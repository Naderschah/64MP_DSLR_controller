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
using TiffImages
using Images
using ImageTransformations
using JLD2 # Saving data
using Random
using Statistics

function Generate_Offsets(px_shift, img1, img2, grey_project::String, match_method="CCORR")
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

    # Generate cropped image and its new shift relative to original image
    img2, shift_in_crop = GenerateMatchCropAndShift(img2, 0.6, px_shift, 0.1)
    # Do greyscale
    img2 = projector(img2)
    img1 = projector(img1)

    # Do the matching algorithm
    return DoMatching(img1, img2,"CCORR", shift_in_crop, [99,99])
end



function DoMatching(image, crop,method, presumed, variation=[99,99])
    #=
    This function takes the an image and a crop of the image to be matched and shifts the second relative to the first to find a match according to the method passed
    presumed is the 2 by 1 array containing the x and y shift (in px) assumed from the relative position
    variation gives the maximum bounds of variation of the image (where we go half the value in each direction) -> We dont expect a large deviation so we limit the search to save time
    =#
    if !all(y->isodd(y), variation)
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
    crop_half_dim = [floor(Int, i) for i in size(crop) ./ 2]
    image_size = [floor(Int, i) for i in size(image) ./ 2] # half size to work relative to center
    for i in -floor(Int, variation[1]/2):floor(Int,variation[1]/2) 
        for j in -floor(Int,variation[2]/2):floor(Int,variation[2]/2)
            # Index images to the same size as crop and pass to matcher with different pixel shifts
            results[i+floor(Int, variation[1]/2)+1,j+floor(Int,variation[2]/2)+1] = mean(matcher(image[image_size[1]-crop_half_dim[1]+i:image_size[1]+crop_half_dim[1]+i, image_size[2]-crop_half_dim[2]+j:image_size[2]+crop_half_dim[2]+j], crop))
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
    crop_vol = comparison_vol / 2 .* [i for i in size(to_crop)][1:2]
    outward = [trunc(Int,i) for i in crop_vol]
    shift = [trunc(Int, i) for i in shift]
    # Crop image within comparison_vol
    to_crop = to_crop[Int(shift[1]-outward[1]):Int(shift[1]+outward[1]), Int(shift[2]-outward[2]):Int(shift[2]+outward[2]), :]

    return to_crop, shift
end




function GeneratePxCoordinates(mm_shift, px_size, magnification, )
    #=
    Function to determine the expected shift between image centers
    =#

    return mm_shift ./ (px_size/magnification)
end


function PyLoadTiff(path)
    imageio = pyimport("imageio")
    np = pyimport("numpy")
    im = np.rot90(imageio.imread(path))  # 4th channel is alpha channel and numpy arrays are immutable
    return im[:,:,1:3]
end

function MakeMosaik_UsingCentralPortionOfImage(image_directory::String, mm_step=0.00012397) # Add image rotation add argument for y and z systematic shift
    #= 
    Takes the relative positions focus map and creates a mosaik
    For now loads images with python cause of applied COMPRESSION_LZW not being implemented in julia TiffImages
    Once pipeline is fully julia pure julia loading can be used again
    =#
    files = readdir(image_directory)
    y_z = [split(i, "_") for i in files]
    y_z = [[parse(Int, String(i[1])), parse(Int, String(split(i[2], ".")[1]))] for i in y_z]
    # Get y and z coordinates
    y = unique([i[1] for i in y_z])
    z = unique([i[2] for i in y_z])
    y = sort(y)
    z = sort(z)
    # We get img size to allocate the array later when we know the dtype
    len_y = Int((length(y)+3)*0.5*3040)
    len_z = Int((length(z)+3)*0.5*4056)
    im_y = 3040
    im_z = 4056
    # We need cumulative shifts for each image 
    shifts = zeros(Int,length(z), length(y),2)
    # Initialize array so its scope is outside the for loop
    findat = zeros(Int,1, 1,2)
    # Images will be made at -90 deg rotation so we load rotate and match 
    # 0,0 corresponds to the top right corner of the image
    # We iterate z then Y
    first = true
    println(z)
    println(y)
    println(files)
    for i in 1:length(z)
        for j in 1:length(y)
            println("$(image_directory)$(y[j])_$(z[i]).tiff")
            if false # TODO Once pure julia this can work just dont apply the compression LZW
                img1 = TiffImages.load("$(image_directory)$(y[j-1])_$(z[i]).tiff")
                print(permutedims(dropdims(channelview(img1),4),(2,3,1)))
                img1 = imrotate(permutedims(dropdims(channelview(img1),4),(2,3,1)), -0.5*pi)
                img2 = TiffImages.load("$(image_directory)$(y[j])_$(z[i]).tiff")
                img2 = imrotate(permutedims(dropdims(channelview(img2),4),(2,3,1)), -0.5*pi)
            end
            skip = true # TODO : Debug variable
            if (i != 1 & j != 1) # If any image inside grid not edge
                img1 = PyLoadTiff("$(image_directory)$(y[j-1])_$(z[i]).tiff")
                img2 = PyLoadTiff("$(image_directory)$(y[j])_$(z[i]).tiff")
            elseif (i == 1 & j == 1) # If first image
                img2 = PyLoadTiff("$(image_directory)$(y[j])_$(z[i]).tiff")
                # Skip checking of shifts
                skip = true
            elseif (i == 1 & j != 1) # If image in first row, same behavior as any image in grid - included for verbosity
                img1 = PyLoadTiff("$(image_directory)$(y[j-1])_$(z[i]).tiff")
                img2 = PyLoadTiff("$(image_directory)$(y[j])_$(z[i]).tiff")
            elseif (i!= 1 & j == 1) # If first image of row we check difference relative to previous first image
                img1 = PyLoadTiff("$(image_directory)$(y[j])_$(z[i-1]).tiff")
                img2 = PyLoadTiff("$(image_directory)$(y[j])_$(z[i]).tiff")
            end # Dont need to check for endbounds thats handeld in the image substitution below
            # Generate absolute coordinates
            if i != 0
                z_max = trunc(Int, GeneratePxCoordinates([z[i] * mm_step, 0], 1.55e-3, 2)[1])
            else
                z_max = 0 
            end
            if j != 0 
                y_max = trunc(Int, GeneratePxCoordinates([0, y[j] * mm_step], 1.55e-3, 2)[2])
            else
                y_max = 0
            end
            # Since y is inverted
            y_max = - y_max


            if !skip # Check shift relative to last image (img1)
                # Z shift is zero here
                match_array = Generate_Offsets(px_shift, img1, img2, "l-star")
                # We find the maximum match_array and compute the corresponding shift
                z_max_temp, y_max_temp = Tuple(argmax(match_array)) .- [floor(Int, i) for i in size(match_array) ./ 2]
                # We add to each following element so that the shift is stored for future shifts
                # Column z at index y for value z 
                shifts[i:end, j, 1] .+= z_max_temp + trunc(Int, px_shift[1])
                # ROw y at index z 
                shifts[i, j:end, 2] .+= y_max_temp + trunc(Int, px_shift[2])
                # Extract current shift
                z_max, y_max = shifts[i,j,:]
                # Since the image starts in the top right
                z_max, y_max = z_max, -y_max
            end

            if first
                # Allocate array to hold results
                first = false
                # Initialize with 100 px extra in each axis so taht an outward shift can be done
                findat = Array{eltype(img2)}(undef,  len_z+100, len_y+100, 3)
                # Append first image - first index is column second is row
                #findat[1+len_z - im_z  : len_z ,  1+len_y- im_y  : len_y,:] .= img1
                println("Made new array")
            end
            
            # We select the bounds
            # First for z 
            # Var to describe column shift top position of image
            # add a constant offset so no out of bounds occurs, add 3/4s image width for the first image/column add image shift computed in px and the adjusted shift (from shifts array computed above)
            z_pos = floor(Int, 100 + (3*im_z/4) +z_max- (i-1)*274) # + (i)*im_z/2 
            if i == 1  # If first row (top row)
                z_low = floor(Int, z_pos)
                z_high = floor(Int, z_pos + 3*im_z/4)
                z_im_low = floor(Int, 1*im_z/4)
                z_im_high = im_z
            elseif  i == length(z) # If last row (bottom row)
                z_low = floor(Int, z_pos - 1*im_z/4)
                z_high = floor(Int, z_pos + im_z/2)
                z_im_low = floor(Int, im_z/4)
                z_im_high = im_z
            else # If in between row
                z_low = z_pos
                z_high = floor(Int, z_pos + (im_z/2))
                z_im_low =floor(Int, im_z/4)
                z_im_high =floor(Int, 3*im_z/4)
            end
            # Now y 
            # Var to describe row left corner position of image
            # length of image + the computed shift - the first image offset of 1/4imsize - standard width of images already added 
            y_pos = floor(Int, len_y+y_max - 3*im_y/2 + (j-1)*133)   # (1*im_y/4) - (j)
            if j == 1 # Very right
                y_low = y_pos # Since this already accounts for the first image offset
                y_high = floor(Int, y_pos + 3*im_y/4) 
                y_im_low = floor(Int, 1*im_y/4)
                y_im_high = im_y
            elseif j == length(y) # Very left so 3/4ths of image in picture
                y_low = floor(Int, y_pos - 1*im_y/4)
                y_high = floor(Int, y_pos + im_y/2)
                y_im_low = 1
                y_im_high = floor(Int, 3*im_y/4)+1 # TODO : Addition of 1 feels wrong but im dim were wrong I think on the low bound i need to add 1 but then it should have raised an error with the other images as well
            else # In the column
                y_low = y_pos 
                y_high = floor(Int, y_pos + im_y /2 )
                # Select middle section
                y_im_low = floor(Int, im_y/4)
                y_im_high = floor(Int, 3*im_y/4)
            end
            # And now we add the image to the Array
            println("Fin Image pos")
            println([z_low,  z_high , y_low, y_high])
            print("Working Image pos")
            println([z_im_low,z_im_high, y_im_low,y_im_high])
            findat[z_low : z_high , y_low : y_high , :] .= img2[z_im_low:z_im_high, y_im_low:y_im_high,:]
        end
        save("Test_$i.png",findat)
    end
   return findat 
end


# final = MakeMosaik_UsingCentralPortionOfImage("/home/felix/hugin_processing/focused/")
# save_object("Combined_Image.jld2", final)
# using Images
# save("Combined_Image.png",final)