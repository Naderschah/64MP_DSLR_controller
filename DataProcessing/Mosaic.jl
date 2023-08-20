#=
All code regarding mosaiking the images will live here



The method I will follow, is to use the known spatial position relative to one another

and then finetune it using an inverse FFT on the known overlap in greyscale (ie shrink array beforehand) 
then we perform the complex conjugate on the FFT, apply the inverse FFT to return to the spatial domain and find the shift best aligning the images


=#
using Primes # This is for convolve inplace
using FFTW # This is for convolve inplace
include("convolve_inplace.jl") # Include for cross correlation function
include("Get_Focus.jl") # Include to get access to greyscale


function Generate_Offsets(mm_shift, img1, img2, grey_project::String, match_method="CCORR")
    #=
    Overarching functions to take images and generate a list with the relative positions to one another for each image
    =#

    # Get greyscale func -> syntax wrong TOOD
    if grey_project == "l-star"
        projector = Grey_Project_l_star
    elseif grey_project == "average"
        projector = Grey_Project_Average
    else
        throw("Projection method $grey_project not implemented")
    end

    # Generate the relative offset in pixels TODO
    shift = GeneratePxCoordinates(mm_shift=mm_shift, px_size=1.55e-3, magnification=2)

    # Generate cropped image and its new shift relative to original image
    img2, shift_in_crop = GenerateMatchCropAndShift(img2, overlap=0.6, shift=shift, comparison_vol=0.1)
    # Do greyscale
    img2 = projector(img2)
    img1 = projector(img1)

    # Do the matching algorithm
    return DoMatching(img1, img2,method="CCORR", presumed=shift_in_crop)
end



function DoMatching(image, crop,method, presumed::Array{2,1}, variation=[100,100])
    #=
    This function takes the an image and a crop of the image to be matched and shifts the second relative to the first to find a match according to the method passed
    presumed is the 2 by 1 array containing the x and y shift (in px) assumed from the relative position
    variation gives the maximum bounds of variation of the image (where we go half the value in each direction) -> We dont expect a large deviation so we limit the search to save time
    =#
    if !all(y->iseven(y), variation)
        throw("Variation must be even")
    end

    shift = [0,0]
    # Array that will hold result of match
    results = Array{Float64}(undef, variation[1], variation[2])

    if method == "CCORR"
        matcher =  normxcorr2_preallocated
    elseif method == "CCOEFF"
        throw("")
    else
        throw("Matching method $method not implemented")
    end
    # Iterate varaition parameters
    crop_half_dim = size(crop) ./ 2
    image_size = size(image)
    for i in 1:variation[1]
        for j in 1:variation[2]
            # Index images to the same size as crop and pass to matcher
            results[i][j] = matcher(image[image_size[1]-crop_half_dim[1]:image_size[1]+crop_half_dim[1], image_size[2]-crop_half_dim[2]:image_size[2]+crop_half_dim[2]], crop)
        end
    end
    # Return the match results
    return results
end



function GenerateMatchCropAndShift(to_crop, overlap, shift, comparison_vol)
    #=
    Generates a cropped image to be matched and returns the relative pixel shift in the new crop of the new center
    to_crop --> the image to crop
    overlap --> The overlap specified during imaging
    comparison_vol --> final crop of image width and height as a percentage
    =#
    # First figure out if we are shifting left or right and up or down
    if shift[1] < 0
        # Crop left portion
        mid_crop_x = trunc(Int,(1+trunc(Int,size(to_crop, 1)*overlap))/2)
    else
        # Crop right portion
        mid_crop_x = trunc(Int,(size(to_crop, 1)*(1-overlap) + size(to_crop, 1)) / 2)
    end
    # Y coordinate wrt image go top down but grid goes bottom up
    if shift[2] < 0
        mid_crop_y = trunc(Int,(size(to_crop, 2)*(1-overlap) + size(to_crop, 2)) / 2)
    else
        mid_crop_y = trunc(Int,(1+trunc(Int,size(to_crop, 2)*overlap))/2)
    end
    # Adjust center shift coordinates
    shift = shift .+ [mid_crop_x, mid_crop_y]
    # Outward shift of midpoint
    outward = trunc(Int,comparison_vol / 2 .* size(to_crop))
    # Crop image within comparison_vol
    to_crop = to_crop[shift[1]-outward:shift[1]+outward, shift[2]-outward:shift[2]+outward]

    return to_crop, shift
end




function GeneratePxCoordinates(mm_shift, px_size, magnification, )
    #=
    Function to determine the expected shift between image centers
    =#

    return mm_shift ./ (px_size/magnification)
end




function MakeMosaik()
    #= 
    Takes the relative positions focus map and creates a mosaik
    =#
end