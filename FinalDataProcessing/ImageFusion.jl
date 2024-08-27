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

function pad_img(arr, pad_w, pad_h)
    # Dimensions of the original array
    w, h, c = size(arr)

    # New dimensions after padding
    new_w = w + 2 * pad_w
    new_h = h + 2 * pad_h
    
    # Creating a new array with the same type as the original
    padded_arr = similar(arr, new_w, new_h, c)
    
    # Copying the original array into the center of the new padded array
    padded_arr[pad_w+1:end-pad_w, pad_h+1:end-pad_h, :] .= arr
    
    # Pad the left and right columns
    padded_arr[1:pad_w, pad_h+1:end-pad_h, :] .= repeat(arr[ 1:1, :, :], pad_w, 1, 1)
    padded_arr[end-pad_w+1:end, pad_h+1:end-pad_h, :] .= repeat(arr[end:end, :, :], pad_w, 1, 1)

    # Now pad the top and bottom using the already padded columns
    padded_arr[:, 1:pad_h, :] .= repeat(padded_arr[:, pad_h+1:pad_h+1, :], 1, pad_h, 1)
    padded_arr[:, end-pad_h+1:end, :] .= repeat(padded_arr[:, end-pad_h:end-pad_h, :], 1, pad_h, 1)
    
    return padded_arr
end

# Merten Kautz van Reeth image fusion
function MKR(fnames, pp::Main.Datastructures.ProcessingParameters, epsilon=1e-10; raw=true, filter=MKR_functions.Pyramid_Filter(), apply_corrections=false)
    # N Images
    N = size(fnames,1)
    # Number of pyramid levels
    nlev = floor(log(min(pp.height ,pp.width)) / log(2))
    padding = 4
    width, height = pp.width + 2*padding,pp.height+ 2*padding
    # Prealocate data structures
    pyr, pyr_Weight, Weight_mat = MKR_functions.GenerateEmptyPyramids(width, height, nlev, N)

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
                    img = MKR_functions.SimpleDebayer(img)  # Out :=> 2028,1520,3 UInt16 but act UInt12
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
        Weight_mat[:,:,:,x] = pad_img(pp.ContrastFunction(img, pp), padding, padding)
        # Generate image Pyramid
        if filter == nothing
            img_pyr = MKR_functions.Laplacian_Pyramid(pad_img(img, padding, padding), nlev)
        else
            img_pyr = MKR_functions.Laplacian_Pyramid(pad_img(img, padding, padding), nlev, filter=filter)
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
    # Clamp and remove padding
    res = clamp.(res, 0, 1)[padding+1:end-padding, padding+1:end-padding, :]
    return res
end



end # module