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
include("PostProcessing.jl")
import .PostProcessing


using FixedPointNumbers
using HDF5


# Merten Kautz van Reeth image fusion
function MKR(fnames, pp::Main.Datastructures.ProcessingParameters, epsilon=1e-10; raw=true, filter=MKR_functions.Pyramid_Filter(), apply_corrections=false)
    # N Images
    N = size(fnames,1)
    # Number of pyramid levels
    nlev = floor(log(min(pp.height ,pp.width)) / log(2))
    # Prealocate data structures
    pyr, pyr_Weight, Weight_mat = MKR_functions.GenerateEmptyPyramids(pp.width,pp.height, nlev, N)

    Threads.@threads for x in eachindex(fnames)
        # Preallocate in case of error
        img = zeros(UInt8, pp.width, pp.height, 3)
        norm = 1
        try # In case some file is damaged
            # Load image
            if fnames[x][end-3:end] == ".dng"
                img = IO_dp.LoadDNGLibRaw(fnames[x], (3,pp.width,pp.height))
                norm = typemax(eltype(img))
            elseif fnames[x][end-3:end] == ".png" 
                img = permutedims(Images.channelview(Images.load(fnames[x]))[1:3,:,:], (2,3,1))
                img = img/typemax(eltype(img))
                norm = 1
            elseif fnames[x][end-4:end] == ".hdf5"
                _file = HDF5.h5open(fnames[x], "r")
                img = HDF5.read(_file["image"])
                if raw#read(_file["image"]["stream"]) == "raw" # The stream attribute was missing form one image
                    # Img data is stored as double width UInt8 array
                    img = reinterpret(UInt16,img[1:end-16,:]) # Want 4056, 3040
                    # Debayer
                    img = SimpleDebayer(img)  # Out :=> 2028,1520,3 UInt16 but act UInt12
                    img = img ./ (2^12-1) # Float64 0 -> 1
                    #img = PostProcessing.apply_ccm(img) # Float64 0 -> 1
                    #img = PostProcessing.apply_gamma_correction(trunc.(UInt16,img.*(2^16-1))) # Uint16 with appropriate data range
                    norm = 1
                else
                    img = permutedims(img, (3,2,1))
                    norm = typemax(eltype(img)) # TODO: May be wrong
                end
                HDF5.close(_file)
            else
                error("Image format not known")
            end
        catch e 
            printstyled("Failed Loading Image, or preprocessing\nError:\n", color=:red)
            println(e)
        end

        if apply_corrections
            # Remove blackpoint (uint16)
            img = ScalingFunctions.removeBlackpoint(img, pp.blackpoint/(2^16-1))
            # Apply flat field, just straightforward division of each pixel 
            img = img ./ permutedims(pp.flat, (2,3,1))
            # Now CCM, for CCM we first need to normalize the color for best results
            h,w,c = size(img)
            px_magnitude = sum(img, dims=3)
            # Normalize the pixel values
            img = img./px_magnitude
            # Apply CCM
            img = reshape( reshape(img, h*w, 3) * pp.CCM , h,w,3)
            # And restore magnitude
            img = img .* px_magnitude
            # Now we clip in case any values went outside the expected range
            img = clamp.(img, 0, 1)
        end

        # Scale 0 to 1, julia does math in 32 bit even if we set 16, so 32 bit float
        img = Float32.(img ./ norm)
        if any(isnan.(img))
            println("Nan in $(fnames[x]). NaN percentage $(sum(isnan.(img))/length(img))")
            # Replace nans with 0s
            replace!(img, NaN=>0)
        end
        # Compute Contrast
        Weight_mat[:,:,:,x] = pp.ContrastFunction(img, pp)
        # Generate image Pyramid
        if filter == nothing
            img_pyr = MKR_functions.Laplacian_Pyramid(img, nlev)
        else
            img_pyr = MKR_functions.Laplacian_Pyramid(img, nlev, filter=filter)
        end
        # Assign to final pyramid
        for l in (1:nlev) @inbounds pyr[l][:,:,:,x] = img_pyr[l]  end
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
        if filter == nothing
            tmp = MKR_functions.Gaussian_Pyramid(Weight_mat[:,:,:,i])
        else
            tmp = MKR_functions.Gaussian_Pyramid(Weight_mat[:,:,:,i],filter=filter)
        end
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

    #Threads.@threads for l in (1:nlev)
    #    @inbounds fin_pyr[l] = sum(pyr_Weight[l][:,:,:,:] .* pyr[l][:,:,:,:], dims=4)[:,:,:,1]  
    #end

    
    # Reconstruct image and return
    if filter == nothing
        res = MKR_functions.Reconstruct_Laplacian_Pyramid(fin_pyr)
    else
        res = MKR_functions.Reconstruct_Laplacian_Pyramid(fin_pyr, filter=filter)
    end
    if maximum(res) > 1
        res ./= maximum(res)
    end
    res = clamp.(res, 0, 1)
    return res
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