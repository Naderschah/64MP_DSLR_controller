# Code here pertains to locating the images based on the step sizes magnification etc, in the final image

module ImagePlacement

include("Datastructures.jl")
import .Datastructures


function GeneratePxCoordinates(pos, ImagingParams::Datastructures.ImagingParameters)
    #Function to determine the expected position of images
    return pos .* ImagingParams.steps_per_mm ./ (ImagingParams.px_size/ImagingParams.magnification)
end

function GenerateFinalArray(ImagingParams, img_name_grid)
    #=
    Here we generate the final array of images to be used for the final image
    =#
    im_width, im_height = size(img_name_grid) .+ 1
    
    f_height = ImagingParams.im_height * ImagingParams.overlap * im_height
    f_width = ImagingParams.im_width * ImagingParams.overlap * im_width
    f_height = trunc(Int,f_height)+10
    f_width = trunc(Int,f_width)+10
    final_array = zeros(f_height,f_width,3)
    return final_array
end






end