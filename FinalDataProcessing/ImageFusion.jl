module ImageFusion

import Images
include("Datastructures.jl")
import .Datastructures
include("MKR_functions.jl")
import .MKR_functions
include("./IO_dp.jl")
import .IO_dp
include("ScalingFunctions.jl")
import .ScalingFunctions

using FixedPointNumbers
using HDF5

"""
Raw data needs to be repacked
3 x 8bit values correspond to 2 x 12 bit values that we save as Uint16
TODO: THis is no longer relevant replace with just reinterpreting 2uint8 as 1uint16 adn then rescale from 12 to 16
"""
function unpack_12bit_packed(data, width, height)
    # Calculate the new length of the data array after removing extra pixels
    # Assuming data is a flat 1D array of bytes
    new_width = Int((width - 28)/1.5)  # Adjust based on actual extra pixels

    # Prepare an array to hold the unpacked 16-bit pixel values
    unpacked = Array{UInt16}(undef, new_width * height)

    # Index for unpacked array
    j = 1

    # Process every 3 bytes to get 2 pixels
    for i in 1:3:length(data) - 2
        # First 12-bit pixel
        first_pixel = (UInt16(data[i]) << 4) | (UInt16(data[i + 1] & 0xF0) >> 4)
        unpacked[j] = first_pixel
        j += 1

        # Second 12-bit pixel
        second_pixel = (UInt16(data[i + 1] & 0x0F) << 8) | UInt16(data[i + 2])
        unpacked[j] = second_pixel
        j += 1

        # Break if the array is filled
        if j > new_width * height
            break
        end
    end

    # Reshape the unpacked array to the correct dimensions
    unpacked = reshape(unpacked, new_width, height)
    # And rescale to uint 16
    return  UInt16.(((unpacked ./ (4095)) .* typemax(UInt16) .÷ 1)) 
end


# Merten Kautz van Reeth image fusion
function MKR(fnames, pp::Main.Datastructures.ProcessingParameters, epsilon=1e-10)

    # N Images
    N = size(fnames,1)
    # Number of pyramid levels
    nlev = floor(log(min(pp.height ,pp.width)) / log(2))
    # Prealocate data structures
    pyr, pyr_Weight, Weight_mat = MKR_functions.GenerateEmptyPyramids(pp.height,pp.width, nlev, N)

    Threads.@threads for x in eachindex(fnames)

        # Load image
        if fnames[x][end-3:end] == ".dng"
            img = IO_dp.LoadDNGLibRaw(pp.path*fnames[x], (3,pp.height,pp.width))
        elseif fnames[x][end-3:end] == ".png" 
            img = permutedims(Images.channelview(Images.load(pp.path*fnames[x]))[1:3,:,:], (3,2,1))
        elseif fnames[x][end-4:end] == ".hdf5"
            _file = HDF5.h5open(pp.path*fnames[x], "r")
            img = HDF5.read(_file["image"])
            if read(_file["image"]["stream"]) == "raw"
                # We need to remove the 28 inactive px in width 
                #img = img[14:end-14, :]
                # And repack, each number is 12 bit, but 3 are packed into two values
                img = unpack_12bit_packed(img, size(img,1), size(img,2))
                # Debayer
                img = SimpleDebayer(img)  # Out :=> 2028,1520,3
            else
                img = permutedims(img, (3,2,1))
            end
            HDF5.close(_file)
            
        else
            println("Image format not known")
        end
        # Remove blackpoint
        img = ScalingFunctions.removeBlackpoint(img, pp.blackpoint)
        # Scale 0 to 1
        img = img ./ typemax(eltype(img))
        if any(isnan.(img))
            println("Nan in $(fnames[x]). NaN percentage $(sum(isnan.(img))/length(img))")
            println("Replacing nans with 0s")
            # Replace nans with 0s
            replace!(img, NaN=>0)
        end
        # Compute Contrast
        Weight_mat[:,:,:,x] = pp.ContrastFunction(img, pp)
        # Generate image Pyramid
        img_pyr = MKR_functions.Laplacian_Pyramid(img, nlev)
        # Assign to final pyramid
        for l in (1:nlev) @inbounds pyr[l][:,:,:,x] = img_pyr[l]  end

        println("Added image $(fnames[x])")

    end
    # If nans restart with higher epsilon
    if any(isnan.(Weight_mat))
        println("Nan in weight matrix")
        return Weight_mat
    end
    # Scale weight matrix such that it is in the range 0 to 1 and each pixel adds up to 1
    Weight_mat = ScalingFunctions.ScaleWeightMatrix(Weight_mat, epsilon)

    # Create pyramid weight matrix
    Threads.@threads for i = 1:N
        tmp = MKR_functions.Gaussian_Pyramid(Weight_mat[:,:,:,i])
        for l in (1:nlev)
            pyr_Weight[l][:,:,:,i] = tmp[l]
            if any(isnan.(pyr_Weight[l][:,:,:,i]))
                println("Nan in pyramid weight matrix")
                return pyr_Weight[l]
            end
        end
    end
    # Free weight matrix memory
    Weight_mat = nothing

    # Create final pyramid
    fin_pyr = Dict()
    Threads.@threads for l in (1:nlev)
        @inbounds fin_pyr[l] = sum(pyr_Weight[l][:,:,:,:] .* pyr[l][:,:,:,:], dims=4)[:,:,:,1]  
    end
    
    # Reconstruct image and return
    return clamp.(MKR_functions.Reconstruct_Laplacian_Pyramid(fin_pyr), 0, 1)
end


function SimpleDebayer(img, bayer_patter = "BGGR")
    """Reduces image size by half simply overlapping the pixels"""
    output = Array{UInt16}(undef,(Int(size(img, 1) / 2), Int(size(img, 2)/ 2), 3))
    # TODO: Generalize for differnt bayer orders, or not ltos of operations involved, and this isnt going to change
    for i in 1:2:size(img,1)-1
        for j in 1:2:size(img,2)-1
            # B (Blue)
            output[i ÷ 2 + 1, j ÷ 2 + 1, 3] = img[i, j]
            # G1 (Green)
            output[i ÷ 2 + 1, j ÷ 2 + 1, 2] = (img[i, j+1]÷2 + img[i+1, j]÷2)
            # R (Red)
            output[i ÷ 2 + 1, j ÷ 2 + 1, 1] = img[i+1, j+1]
        end
    end
    return output
end

end # module