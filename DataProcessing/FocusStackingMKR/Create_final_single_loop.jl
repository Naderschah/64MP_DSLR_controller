#=
Here we do the same processing as usual, but we make hdrs at the last step 
Focus maps are computed once from the selected exposure and the final image with focus map is created 

Images are saved per exposure so that hdr can still be done
=#
using JLD2 # Saving data
using Images
using PyCall
include("Get_Focus.jl")
include("Mosaic.jl")

# Paths
image_directory = "/media/felix/Drive/Images/img_1/"
save_path = "/home/felix/rapid_storage_1/Cranefly/"


function Generate_ImCoords_in_final(i, j, im_z, im_y, mm_step, z, y, len_y, len_z)
    # Last term in each pos is for statistical error correction
    
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

    z_pos = floor(Int, (3*im_z/4) +z_max- (i-1)*274) # + (i)*im_z/2    
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
        y_im_high = floor(Int, 3*im_y/4)+1
    else # In the column
        y_low = y_pos 
        y_high = floor(Int, y_pos + im_y /2 )
        # Select middle section
        y_im_low = floor(Int, im_y/4)
        y_im_high = floor(Int, 3*im_y/4)
    end
    return [z_low,  z_high , y_low, y_high, z_im_low,z_im_high, y_im_low,y_im_high]
end

function Generate_z_HalfIm(i, im_z, len_z, num_z)
    """Uses Image half size to compute position"""
    if i == 1 # First image in z (height)
        im_z_start = 1
        im_z_stop = floor(Int, 3/4 * im_z)
        z_im_low = 1
        z_im_high = im_z_stop
    elseif i == num_z
        #             nr images * size /2 + half size added by top and bottom
        im_z_stop =  floor(Int,(im_z + 1) / 2 * num_z )
        im_z_start = floor(im_z_stop - 3/4*im_z)
        z_im_low = floor(Int, im_z/4)
        z_im_high = im_z
    else
        #             nr images * size /2 + quarter size added by top
        im_z_stop =  floor(Int,(im_z + 1/2)/ 2 * num_z ) 
        im_z_start = floor(Int, im_z_stop - 1/2*im_z)
        z_im_low =floor(Int, im_z/4)
        z_im_high =floor(Int, 3*im_z/4)
    end # Also apply statistical error
    return [im_z_start- (i-1)*274, im_z_stop- (i-1)*274, z_im_low, z_im_high]
end


function Generate_y_HalfIm(j, im_y, len_y, num_y)
    """Uses Image half size to compute position""" # TODO : Add to func below and test withotu images
    # Note that y is inverted
    if j == 1 # First image in y (width)
        im_y_start = len_y - floor(Int, 3/4 * im_y)
        im_y_stop = len_y 
        y_im_low = floor(Int, 1*im_y/4) # RHS so upper part 
        y_im_high = im_y
    else
        #            length of image - first (overshot) - subsequent
        im_y_stop =   floor(Int, len_y - 1/4 * im_y - im_y/2*(j-1) )  
        im_y_start = floor(Int, im_y_stop - 1/2*im_y)
        y_im_low =floor(Int, im_y/4)
        y_im_high =floor(Int, 3*im_y/4)
    end
    if j == num_y
        im_y_start = im_y_start - floor(Int, im_y/4) + 1 # Start 1/4 earlier
        y_im_low = 1
    end
    return [im_y_start + (j-1)*133, im_y_stop + (j-1)*133, y_im_low, y_im_high]
end


function LoadDNGandRotate(path)
    rawpy = pyimport("rawpy") 
    np = pyimport("numpy") 
    im_1 = rawpy.imread(path)
    im_1=im_1.postprocess(demosaic_algorithm=rawpy.DemosaicAlgorithm(11),half_size=false, 
                                # 3color no brightness adjustment (default is false -> ie auto brightness)
                                four_color_rgb=false,no_auto_bright=true,
                                # If using dcb demosaicing
                                dcb_iterations=0, dcb_enhance=false, 
                                # Denoising
                                fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode(0),noise_thr=nothing,
                                # Color
                                median_filter_passes=0,use_camera_wb=false,use_auto_wb=false, user_wb=nothing,
                                # sRGB output and output bits per sample : 8 is default
                                output_color=rawpy.ColorSpace(1),output_bps=16,
                                # Black levels for the sensor are predefined and cant be modified i think
                                user_flip=nothing, user_black=nothing,
                                # Adjust maximum threshholds only applied if value is nonzero default was 0.75
                                # https://www.libraw.org/docs/API-datastruct.html
                                user_sat=nothing, auto_bright_thr=nothing, adjust_maximum_thr=0, bright=1.0,
                                # Ignore default is Clip
                                highlight_mode=rawpy.HighlightMode(1), 
                                # Exp shift 1 is do nothing, nothing should achieve the same but to be sure, preserve 1 is full preservation
                                exp_shift=1, exp_preserve_highlights=1.0,
                                # V_out = gamma[0]*V_in^gamma 
                                gamma=(2.222, 4.5), chromatic_aberration=(1,1),bad_pixels_path=nothing
                                )
        return np.rot90(im_1)
end

function MakeMainExp(image_directory, save_path)
    # Constants 
    im_y = 3040
    im_z = 4056
    mm_step=0.00012397
    

    # Do file selection from folder
    files = readdir(image_directory)
    x_y_z_exp = [split(i, "_") for i in files]
    x_y_z = [[parse(Int, String(i[1])),parse(Int, String(i[2])), parse(Int, String(i[3]))] for i in x_y_z_exp]
    exp = unique([parse(Int, String(split(i[4], ".")[1])[4:end-3]) for i in x_y_z_exp])
    # Get x, y and z coordinates
    x = unique([i[1] for i in x_y_z])
    y = unique([i[2] for i in x_y_z])
    z = unique([i[3] for i in x_y_z])
    x = sort(x)
    y = sort(y)
    z = sort(z)
    exp = sort(exp)
    println("Exposures")
    println(exp)

    # Get image sizing
    len_y = floor(Int,(length(y)+1)*0.5*3040) 
    len_z = floor(Int,(length(z)+1)*0.5*4056) 
    num_exp = length(exp)
    # Index for middle exposure
    mid_exp_index = trunc(Int, num_exp/2 + 1) 

    # Final image array for each exposure 
    findat = zeros(UInt16,num_exp,len_z, len_y,3)
    # Depthmap array -> subtract 10 to make mark for unfocused
    findat_depthmap = zeros(Int,len_z, len_y) .- 10
    # Fill val array with 0.1 to get a base threshhold for focus identification
    findat_depthmap_val = ones(Float64,len_z, len_y) .* 100
    start_time = time()
    println("Imsize in units im_x/4")
    println("Z: $(len_z / im_z * 4) ; Y: $(len_y/im_y*4)")
    for i in 1:length(z)
        z_low, z_high, z_im_low, z_im_high=Generate_z_HalfIm(i, im_z, len_z, length(z))
        println("Z coords in units im_z/4")
        println("upper $(z_high/im_z*4) im_bounds : $(z_im_low/ im_z*4)::$(z_im_high/ im_z*4)")
        y_start = time()
        for j in 1:length(y)
            println("")
            println("Doing $i from $(length(z)) in z")
            println("Doing $j from $(length(y)) in y")
            # Generate coordinates based on image pos and predicted shift
            y_low, y_high, y_im_low, y_im_high = Generate_y_HalfIm(j, im_y, len_y, length(y))
            println("Y coords in units im_y/4")
            println("$(y_low/im_y*4) ::$(y_high/im_y*4) im_bounds : $(y_im_low/ im_y*4)::$(y_im_high/ im_y*4)")
            y_low, y_high, y_im_low, y_im_high = [trunc(Int, i) for i in (y_low, y_high, y_im_low, y_im_high)]
            # Generate coordinates for the image inside the final image -> based on step counts
            # z_low,  z_high , y_low, y_high,z_im_low,z_im_high, y_im_low,y_im_high = Generate_ImCoords_in_final(i, j, im_z, im_y, mm_step, z, y, len_y, len_z)
            # Generate views of img and depthmap arrays to make further indexing less confusing
            findat_view = @view findat[mid_exp_index, z_low:z_high , y_low:y_high, :]
            findat_depthmap_view = @view findat_depthmap[z_low:z_high , y_low:y_high]
            findat_depthmap_val_view = @view findat_depthmap_val[z_low:z_high , y_low:y_high]
            x_ls = []
            println("Doing Main exposure")
            x_start = time()
            for k in 1:length(x)
                # Load each image in x, compute focus map and assign values based on bounds --> Still need to load with python sadly
                # Img contains crop not full
                img = LoadDNGandRotate("$(image_directory)$(x[k])_$(y[j])_$(z[i])_exp$(exp[mid_exp_index])mus.dng")[z_im_low:z_im_high, y_im_low:y_im_high, :]
                # Compute Focus map check if larger and make assignment (indexing cause of convolution)
                LoG = ComputeFocusMap(img, "l-star")
                bool_array = (LoG .> findat_depthmap_val_view)
                if any(bool_array)
                    # Append x coord to x_ls for later iterating 
                    append!(x_ls, x[k])
                    # Apply x pos to depth map
                    findat_depthmap_view[bool_array] .= x[k]
                    # Apply LoG val to depthmap val array
                    findat_depthmap_val_view[bool_array] .= LoG[bool_array]
                    # Apply image data to image view
                    findat_view[bool_array,:] = img[bool_array,:]
                end
            end
            resid_time =time()
            println("X took $(resid_time-x_start)s")
            println("Doing Residual exposures")
            # Iterate the x vals that were actually in focus
            for k in 1:length(x_ls)
                # Regenerate bool array 
                bool_array = (findat_depthmap_view .== x_ls[k])
                # Now we load the corresponding images for the other exposures and add them to the other layers of the image array
                # And Check this is not the one we did before
                for l in 1:num_exp if (l != mid_exp_index)
                    # Regenerate the view for the corresponding exp index (again for readability)
                    findat_view = @view findat[l, z_low:z_high , y_low:y_high, :]
                    img = LoadDNGandRotate("$(image_directory)$(x_ls[k])_$(y[j])_$(z[i])_exp$(exp[l])mus.dng")[z_im_low:z_im_high, y_im_low:y_im_high, :]
                    findat_view[bool_array,:] = img[bool_array,:]
                    # And repeat
                end end
            end
            println("Precomputed stacking took $(time()-resid_time)s")
            println("So $((time()-resid_time)/(num_exp-1))s per exposure")
        end
        println("Y took $(time()-y_start)s")
        println("Saving intermediary")
        save("$(save_path)FocusedMosaicImage_intermediary.png",findat[mid_exp_index, : ,: ,:])
    end
    println("Stacking took $(time()-start_time)s")
    println("For $(length(files)) files")
    println("Saving images and depthmaps")
    # Now we save everything, as jld2 first in case of exception and then png for each image

    save_object("$(save_path)FocusMosaicedImageMultipExp.jld2", findat)
    save_object("$(save_path)FocusMosaicedDepthmap.jld2", findat_depthmap)
    save_object("$(save_path)FocusMosaicedDepthmapVal.jld2", findat_depthmap_val)
    # Uses Images module to save as png 
    for i in 1:num_exp
        save("$(save_path)FocusedMosaicImage_$(exp[i]).png",findat[i, : ,: ,:]) # exp, height, width, color
    end
end


MakeMainExp(image_directory, save_path)