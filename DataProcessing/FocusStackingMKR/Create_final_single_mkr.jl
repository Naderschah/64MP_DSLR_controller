#=
Here we process all in single loop
Using mericam Kautz van reeth algorithm in a single loop, and hope pc doesnt crash 

Images are saved per exposure so that hdr can still be done


To make video from intermediates:
Set image saving on : TODO Add argument
run in cmd line in folder with appropriate names:
ffmpeg -r 10 -pattern_type sequence -i MKR_focus_%d_%d.png -vcodec mpeg4 -y movie.mp4
-r : rate in Hz
-i : input pattern

=#
using JLD2 # Saving data
using Images
using PyCall
using Dates
using Base.Threads
# Load C code
const LoadFileC = joinpath(@__DIR__, "C_stuff/LibRaw_LoadDNG.so")
# Load Julia code
include("Get_Focus.jl")
include("Mosaic.jl")
include("Mertens_Kautz_Van_Reeth.jl")





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


function LoadDNGLibRaw(path, size=(3,3040,4056),processing_params=Dict())
    # Uses C API LibRaw to load image char array
    # Size should be Color, height, width
    # img is to be populated by the ccall
    img = Array{UInt16}(undef, size[1]*size[2]*size[3])
    success = Ref{Cint}(0)
    @ccall "/home/felix/PiCamera_DSLR_like_controller/DataProcessing/C_stuff/LibRaw_LoadDNG.so".LoadDNG(img::Ref{Cushort},path::Ptr{UInt8},success::Ref{Cint})::Cvoid
    # Grab the reference from mem
    #print("success : $(success[])")
    if success[] == 0
        # TODO: Find a way to actually change the value of success in the C-code for some reason i cant figuer it out
        println("\033[93m   Error loading image : $(path) \033[0m") 
        println(success)
        img .= 0
    end
    # The array is linear so we need to reshape it to get the correct data
    order = [1,3,2]
    size = size[order]
    img = permutedims(reshape(img, size), (3,2,1))

    # Remove blackpoint
    for i in 1:3
        bool_array = img[:,:,i] .<= processing_params["blackpoint"][i]
        img[:,:,i][bool_array] .= processing_params["blackpoint"][i]
        img[:,:,i] .-= processing_params["blackpoint"][i]
    end

    return img
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
                                ) # 
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
                    fname = GenerateFileName(x[k],y[j],z[i],exp[mid_exp_index])
                    img = LoadDNGandRotate("$(image_directory)$fname")[z_im_low:z_im_high, y_im_low:y_im_high, :]
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
                    fname = GenerateFileName(x_ls[k],y[j],z[i],exp[l])
                    img = LoadDNGandRotate("$(image_directory)$fname")[z_im_low:z_im_high, y_im_low:y_im_high, :]
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

function GrabIdentifiers(image_directory)
    files = readdir(image_directory)
    x_y_z_exp = [split(i, "_") for i in files]
    x_y_z = [[parse(Int, String(i[1])),parse(Int, String(i[2])), parse(Int, String(i[3]))] for i in x_y_z_exp]
    exp = []
    try
        exp = unique([parse(Int, String(split(i[4], ".")[1])[4:end-3]) for i in x_y_z_exp])
    catch # For one exposure
        exp = ["NoIR"]
    end
    x = unique([i[1] for i in x_y_z])
    y = unique([i[2] for i in x_y_z])
    z = unique([i[3] for i in x_y_z])
    # We will only do one yz pos for starters
    x = sort(x)
    y = sort(y)
    z = sort(z)
    return x, y, z, exp
end

function LoadDataArray(x, y, z, exp_, im_y, im_z, multip_exp)
    #=
    Loads data in proper formating
    =#
    if multip_exp
        indat = zeros(UInt16,im_z, im_y,length(exp_), 3, length(x))
        for j in 1:length(exp_) 
            for i in 1:length(x)
                fname = GenerateFileName(x[i],y,z,exp_[j])
                im = LoadDNGandRotate("$(image_directory)$fname")
                indat[:,:,j,:,i] = im[:,:,:]
            end 
        end
    else
        indat = zeros(UInt16,im_z, im_y, 3, length(x)*length(exp))
        for i in 1:length(x) 
            for j in 1:length(exp_)
                fname = GenerateFileName(x[i],y,z,exp_[j])
                im = LoadDNGandRotate("$(image_directory)$fname")
                indat[:,:,:,i*j] = im[:,:,:]
            end 
        end
    end
    return indat
end

function DoSave(base_path, naming, data)
    #=
    Saves jld2, does broken save (in case data scale is way off at corners), saves normally
    =#

    # Save Data as jld2 first in case anything is broken
    #save_object("$(base_path)$(naming).jld2", data)

    # Normalize and change dtype to save as png
    min_ = minimum(data)
    max_ = maximum(data)
    # truncate and rescale
    findat_ = trunc.(UInt16,((data .-min_)./(max_.-min_ )) .* (2^16-1)) #.^4.5
    save("$(save_path)$(naming).png", findat_)
    println("Save name $(save_path)$(naming).png")
    # Reload image so its in the right formating and do historgram equalization
    #hist_equal = adjust_histogram(load("$(save_path)$(naming).png"), Equalization(nbins = (2^16-1)))
    #save("$(save_path)$(naming)_histequal.png", hist_equal)
end

function exp_scaling_contrast(x)
    #=
    Function to scale contrast weights,
    LaTeX string: x^{4}\left(e^{x\ln\left(2\right)}-1\right)
    =#
    return x^4*exp(x*log(2))-1
end

function test_MKR_focus_and_HDR(image_directory, save_path, processing_params)
    # Constants 
    im_y = 3040
    im_z = 4056

    
    # Do file selection from folder
    x, y, z, exp_ = GrabIdentifiers(image_directory)

    # For testing purposes select subset
    x = x[trunc(Int, length(x)/2):end]
    
    # Select y and z to be processed (only 1 of each at a time)
    y = y[5]
    z = z[5]

    # Make array to hold input images and populate
    indat = LoadDataArray(x, y, z, exp_, im_y, im_z, processing_params["MultiExp"])

    println("Memory used after image loading")
    println(memuse())
    println("Current Time $(Dates.format(now(), "HH:MM:SS"))")
    # Do exposure fusion NND and whatnot
    findat = Exposure_Fusion(indat, processing_params)
    if any(isnan.(findat)) println("Final Image has $(sum(isnan.(findat))) Nan values") end
    DoSave(save_path, "MKR_focus", findat)
end

# ##### Here is where execution used to be 

# Helper functions for cleaner syntax TODO: Check how julia knows to do float32 math
function cont_STD_helper(dat, prec, args=nothing)
    mean_kernel = ones(Float32, (3,3))
    fe = 1 # Extend of kernel
    cont = prec.(similar(dat))
    for c in (1:3)
        cont[:,:,c] = abs.(imfilter(dat[:,:,c].^2, mean_kernel)[2*fe:end-2*fe-1,2*fe:end-2*fe-1] - imfilter(dat[:,:,c], mean_kernel)[2*fe+1:end-2*fe,2*fe+1:end-2*fe].^2)
    end
    return cont
end


function contrast_LoG(I, prec=float32, grey_project=Grey_lstar)
    #=
    Contrast using Lagrangian of Gaussian
    I -> Image
    grey -> greyscale projector (if requiredd)
    =#
    # Lagrangian of Gaussian 1.4σ
    LoG_kernel = [[0,1,1,2,2,2,1,1,0] [1,2,4,5,5,5,4,2,1] [1,4,5,3,0,3,5,4,1] [2,5,3,-12,-24,-12,3,5,2] [2,5,0,-24,-40,-24,0,5,2] [2,5,3,-12,-24,-12,3,5,2] [1,4,5,3,0,3,5,4,1] [1,2,4,5,5,5,4,2,1] [0,1,1,2,2,2,1,1,0] ]
    kernel_size = trunc(Int, size(LoG_kernel)[1]/2)
    
    return repeat(abs.(DSP.conv(LoG_kernel, grey_project(prec.(I)))[kernel_size+1:end-kernel_size,kernel_size+1:end-kernel_size,:]),outer=[1,1,3])
end


# Redefine some functions for the non variable size implementation
function Reconstruct_Laplacian_Pyramid(pyr)
    #=
    Takes pyramid and makes image
    =#
    nlev = length(pyr)
    # Start with low pass residual
    R = pyr[nlev]
    filter = Pyramid_Filter*Pyramid_Filter'
    for l in ((nlev-1):-1:1)
        # Upsample, add to current level
        odd = 2 .*size(R) .- size(pyr[l])
        R = pyr[l] + upsample(R,odd,filter)
    end
    return R
end

function upsample(I,odd, filter)
    # Make zero array
    r,c = size(I)[1:2].*2
    R =zeros(r,c,3)
    # Get filter array expansion
    # Populate array, for the first we dont need to account for odd
    fe = trunc(Int,size(filter)[1]/2)
    R[1:2:r, 1:2:c, :] = I 
    R[2:2:r, 1:2:c, :] = I 
    R[1:2:r, 2:2:c, :] = I 
    R[2:2:r, 2:2:c, :] = I 
    ## Fix borders in case we need odd sized array
    if odd[1] == 1 R = R[1+odd[1]:end,1:end,:] end
    if odd[2] == 1 R = R[1:end,1+odd[2]:end,:] end
    R = make_padding(R, fe) 
    DSP.conv(R, filter)
    R = make_padding(R, fe) 
    DSP.conv(R, filter)
    multip = 2 
    R = R[multip*fe+1:end-multip*fe, multip*fe+1:end-multip*fe,:]
    return R
end


function FocusFusion_Reduced_Memory(image_directory, save_path, processing_params)
    # Using the entire data array causes a crash so this is used
    # Constants remember camera is rotated
    println("Running with $(nthreads()) threads")
    #im_y = 3040
    #im_z = 4056
    im_z = 3040
    im_y = 4056

    prec = if processing_params["prec32"] float32 else float64 end 

    if !endswith(image_directory, "/") image_directory *= "/" end
    if !endswith(save_path, "/") save_path *= "/" end

    # Do file selection from folder
    x, y, z, exp_ = GrabIdentifiers(image_directory)
    # Grab contrast function 
    if processing_params["ContrastFunction"] == "STD"
        cont_func = cont_STD_helper 
    elseif processing_params["ContrastFunction"] == "LoG" 
        cont_func = contrast_LoG
    end
    # Iterate each seperate image to be done
    for ei in eachindex(exp_)
    for yi in eachindex(y)
    for zi in eachindex(z)
        if !isfile("$(save_path)Focused_y=$(y[yi])_z=$(z[zi])_e=$(exp_[ei]).png")
            #        We start by compressing the images while loading
            # Get the number of images and number of pyramid levels
            N = size(x)[1]
            nlev = floor(log(min(im_z,im_y)) / log(2))
            # We create a dictionary to hold the laplacian transformed data + the later created weight matrix
            pyr = Dict()
            Weight_mat = float32.(zeros(im_z, im_y, 3, N))
            pyr_Weight = Dict()
            # And we populate it with zero arrays
            h_n, w_n = im_y, im_z
            for l in (1:nlev)
                pyr[l] = float32.(zeros(w_n, h_n, 3, N))
                pyr_Weight[l] = Array{Float32}(undef, (w_n, h_n, 3, N))
                w_n2, h_n2 = trunc(Int, w_n/2), trunc(Int, h_n/2)
                if (w_n2 *2 - w_n != 0)  w_n = w_n2+1 
                else w_n = w_n2 end
                if (h_n2 *2 - h_n != 0)  h_n = h_n2+1 
                else h_n = h_n2 end
            end
            # Load Data apply laplacian pyramid and save in dict
            Threads.@threads for xi in eachindex(x) #
                fname = GenerateFileName(x[xi],y[yi],z[zi],exp_[ei])
                im = LoadDNGLibRaw("$(image_directory)$fname", (3,3040,4056), processing_params) # The pyramid reconstruction of this is correct
                im = im ./ typemax(eltype(im))
                
                # Compute contrast
                cont = cont_func(im, prec,processing_params["Greyscaler"]) # Cont here
                #scont = trunc.(UInt8, abs.(cont)./maximum(abs.(cont)) .* (2^8-1))
                #save("$(processing_params["IntermediateSaveLocation"])$(x[xi])_$(y[yi])_$(z[zi])_$(exp_[ei])_cont.png", scont)
                Weight_mat[:,:,:,xi] = cont

                im_pyr = Laplacian_Pyramid(im)
                # assign to pyramid
                for l in (1:nlev)
                    @inbounds pyr[l][:,:,:,xi] = im_pyr[l]
                end
                println("   Added $(image_directory)$fname")
            end
            # Now we need to normalize the weight matrix such that each pixel equals 1, we also set a 0 point
            Weight_mat[:,:,1,:] .-= minimum(Weight_mat[:,:,1,:])
            Weight_mat[:,:,2,:] .-= minimum(Weight_mat[:,:,2,:])
            Weight_mat[:,:,3,:] .-= minimum(Weight_mat[:,:,3,:])
            Weight_mat ./= repeat(sum(Weight_mat,dims=4), outer=[1,1,1,N]) # Normalization is correct sum over im size gives 3 so 1 for each c
            println("Memory used after populating image and weight")
            println(memuse())
            # Now we compute the Gaus pyramid of the weight matrix and add all together
            Threads.@threads for i in 1:N
                tmp = Gaussian_Pyramid(Weight_mat[:,:,:,i])
                #save("$(processing_params["IntermediateSaveLocation"])$(x[i])_$(y[yi])_$(z[zi])_$(exp_[ei])_cont_norm.png", trunc.(UInt8,Weight_mat[:,:,:,i].*255))
                for l in (1:nlev)
                    pyr_Weight[l][:,:,:,i] = tmp[l]
                end
            end
            fin_pyr = Gaussian_Pyramid(zeros(im_z, im_y, 3))
            Threads.@threads for l in (1:nlev)
                @inbounds fin_pyr[l] = sum(pyr_Weight[l][:,:,:,:] .* pyr[l][:,:,:,:], dims=4)[:,:,:,1]  
            end
            printstyled("Saving jld2 pyramid and recreated Images\n", color = :green, bold = true)
            #JLD2.save_object("$(save_path)Focused_y=$(y[yi])_z=$(z[zi])_e=$(exp_[ei])", fin_pyr)

            fin_im = Reconstruct_Laplacian_Pyramid(fin_pyr)
            fin_im = 4.5 .* fin_im.^(1/2.22)
            try
                DoSave(save_path, "Focused_y=$(y[yi])_z=$(z[zi])_e=$(exp_[ei])", fin_im)
            catch
                println("Error saving image : $(save_path)Focused_y=$(y[yi])_z=$(z[zi])_e=$(exp_[ei]).png")
            end
            println("Completed Focused_y=$(y[yi])_z=$(z[zi])_e=$(exp_[ei])")
        else 
            println("Skipping $(save_path)Focused_y=$(y[yi])_z=$(z[zi])_e=$(exp_[ei]).jld2")
        end # If file doesnt exist statement
    end # End for z val loop
    end # End for y val loop
    end # End for exp val loop
end



### Attempt at NND:

function FocusFusion_NND(image_directory, save_path, processing_params)
    # Using the entire data array causes a crash so this is used
    # Constants remember camera is rotated
    println("Running with $(nthreads()) threads")
    #im_y = 3040
    #im_z = 4056
    im_z = 3040
    im_y = 4056

    prec = if processing_params["prec32"] float32 else float64 end 

    if !endswith(image_directory, "/") image_directory *= "/" end
    if !endswith(save_path, "/") save_path *= "/" end

    # Do file selection from folder
    x, y, z, exp_ = GrabIdentifiers(image_directory)
    # Grab contrast function 
    if processing_params["ContrastFunction"] == "STD"
        cont_func = cont_STD_helper 
    elseif processing_params["ContrastFunction"] == "LoG" 
        cont_func = contrast_LoG
    end
    # Iterate each seperate image to be done
    for ei in eachindex(exp_)
    for yi in eachindex(y)
    for zi in eachindex(z)
        if !isfile("$(save_path)Focused_y=$(y[yi])_z=$(z[zi])_e=$(exp_[ei]).png")
            #        We start by compressing the images while loading
            # Get the number of images and number of pyramid levels
            N = size(x)[1]
            nlev = floor(log(min(im_z,im_y)) / log(2))
            # We create a dictionary to hold the laplacian transformed data + the later created weight matrix
            pyr = Dict()
            Weight_mat = float32.(zeros(im_z, im_y, 3, N))
            pyr_Weight = Dict()
            # And we populate it with zero arrays
            h_n, w_n = im_y, im_z
            for l in (1:nlev)
                pyr[l] = float32.(zeros(w_n, h_n, 3, N))
                pyr_Weight[l] = Array{Float32}(undef, (w_n, h_n, 3, N))
                w_n2, h_n2 = trunc(Int, w_n/2), trunc(Int, h_n/2)
                if (w_n2 *2 - w_n != 0)  w_n = w_n2+1 
                else w_n = w_n2 end
                if (h_n2 *2 - h_n != 0)  h_n = h_n2+1 
                else h_n = h_n2 end
            end
            # Load Data apply NNP then laplacian pyramid and save in dict
            # We load the first two images

            fname = GenerateFileName(x[1],y[yi],z[zi],exp_[ei])
            old_im = LoadDNGLibRaw("$(image_directory)$fname", (3,3040,4056), processing_params) # The pyramid reconstruction of this is correct
            old_im = old_im ./ typemax(eltype(old_im))
            old_cont = cont_func(old_im, prec) # Cont here
            old_cont ./= maximum(old_cont)

            fname = GenerateFileName(x[2],y[yi],z[zi],exp_[ei])
            curr_im = LoadDNGLibRaw("$(image_directory)$fname", (3,3040,4056), processing_params) # The pyramid reconstruction of this is correct
            curr_im = curr_im ./ typemax(eltype(curr_im))
            curr_cont = cont_func(curr_im, prec) # Cont here
            curr_cont ./= maximum(curr_cont)

            next_im = []

            # We load the next image but work on the previous one, such that the last image and first image
            # are never added to any array
            alpha=0.4
            for xi in 3:length(x) #
                next_im = LoadDNGLibRaw("$(image_directory)$fname", (3,3040,4056), processing_params) # The pyramid reconstruction of this is correct
                next_im = next_im ./ typemax(eltype(next_im))
                # Compute contrast
                next_cont = cont_func(next_im, prec) # Cont here
                next_cont ./= maximum(next_cont) # Cont here
                # Get kernel and apply NND
                sigma_estimate = NoiseEstimate(curr_im)
                fe = 4 # Filter extent
                filter = GaussianKernel(2*fe+1, sigma_estimate)
                operating_im = curr_im * (1+alpha) .- (old_cont .* abs.(imfilter(old_im, filter, "replicate")[2*fe+1:end-2*fe,2*fe+1:end-2*fe,:]) .+ next_cont .* abs.(imfilter(next_im, filter, "replicate")[2*fe+1:end-2*fe,2*fe+1:end-2*fe,:])) .* (alpha/2)

                # We now have the current image modified and it is ready
                # We need to refresh our 3 images and then we can move on
                old_im = curr_im
                old_cont = curr_cont
                curr_im = next_im
                curr_cont = next_cont

                # Compute NND modified contrast
                cont = cont_func(operating_im, prec) 
                Weight_mat[:,:,:,xi] = cont

                im_pyr = Laplacian_Pyramid(operating_im)
                # assign to pyramid
                for l in (1:nlev)
                    @inbounds pyr[l][:,:,:,xi] = im_pyr[l]
                end
                println("   Added $(image_directory)$fname")
            end
            old_im = []
            curr_im = []
            next_im = []
            old_cont = []
            curr_cont = []
            next_cont = []

            # Now we need to normalize the weight matrix such that each pixel equals 1, we also set a 0 point
            Weight_mat[:,:,1,:] .-= minimum(Weight_mat[:,:,1,:])
            Weight_mat[:,:,2,:] .-= minimum(Weight_mat[:,:,2,:])
            Weight_mat[:,:,3,:] .-= minimum(Weight_mat[:,:,3,:])
            Weight_mat ./= repeat(sum(Weight_mat,dims=4), outer=[1,1,1,N]) # Normalization is correct sum over im size gives 3 so 1 for each c
            println("Memory used after populating image and weight")
            println(memuse())
            # Now we compute the Gaus pyramid of the weight matrix and add all together
            Threads.@threads for i in 1:N
                tmp = Gaussian_Pyramid(Weight_mat[:,:,:,i])
                #save("$(processing_params["IntermediateSaveLocation"])$(x[i])_$(y[yi])_$(z[zi])_$(exp_[ei])_cont_norm.png", trunc.(UInt8,Weight_mat[:,:,:,i].*255))
                for l in (1:nlev)
                    pyr_Weight[l][:,:,:,i] = tmp[l]
                end
            end
            println("Creating final pyramid")
            fin_pyr = Gaussian_Pyramid(zeros(im_z, im_y, 3))
            Threads.@threads for l in (1:nlev)
                @inbounds fin_pyr[l] = sum(pyr_Weight[l][:,:,:,:] .* pyr[l][:,:,:,:], dims=4)[:,:,:,1]  
            end
            printstyled("Saving jld2 pyramid and recreated Images\n", color = :green, bold = true)
            #JLD2.save_object("$(save_path)Focused_y=$(y[yi])_z=$(z[zi])_e=$(exp_[ei])", fin_pyr)

            fin_im = Reconstruct_Laplacian_Pyramid(fin_pyr)
            fin_im = 4.5 .* fin_im.^(1/2.22)
            try
                DoSave(save_path, "Focused_y=$(y[yi])_z=$(z[zi])_e=$(exp_[ei])", fin_im)
            catch
                println("Error saving image : $(save_path)Focused_y=$(y[yi])_z=$(z[zi])_e=$(exp_[ei]).png")
            end
            println("Completed Focused_y=$(y[yi])_z=$(z[zi])_e=$(exp_[ei])")
        else 
            println("Skipping $(save_path)Focused_y=$(y[yi])_z=$(z[zi])_e=$(exp_[ei]).jld2")
        end # If file doesnt exist statement
    end # End for z val loop
    end # End for y val loop
    end # End for exp val loop
end

function NoiseEstimate(Img)
    #=
    Makes an estimate of the noise in the image (used for Nearest neighbor deconvolution)
    Reference: J. Immerkær, “Fast Noise Variance Estimation”, Computer Vision and Image Understanding, Vol. 64, No. 2, pp. 300-302, Sep. 199
    =#
    M = [[1,-2,1] [-2,4,-2] [1,-2,1]]
    fe = 1
    H,W = size(Img)[1:2]
    Img = Grey_Mean(Img)
    sigma = sum(abs.(imfilter(Img, M, "replicate")[2*fe+1:end-2*fe,2*fe+1:end-2*fe,:]))
    return sqrt(sigma * (0.5 * pi)^0.5 / (6 * (W-2) * (H-2)))
end

function GaussianKernel(n, sigma)
    # Create symmetric gaussian kernel of size nxn
    mid = trunc(Int, (n)/2 )
    D1kern = [exp(-(i - mid)^2/(2*sigma^2)) for i in 1:n]
    D1kern ./= sum(D1kern)
    return D1kern*D1kern'
end

#   Back to normal code (above NND)


function Grey_No_red(I)
    return (I[:,:,2] + I[:,:,3])./2
end

function GenerateFileName(x,y,z,exp)
    if typeof(exp) == Int
        return "$(x)_$(y)_$(z)_exp$(exp)mus.dng"
    else
        return "$(x)_$(y)_$(z)_$(exp).dng"
    end
end




start = time()

# Paths
image_directory = "/media/felix/3677421f-5daf-4ea7-ba33-31b09d11edcf/Images/img_0/"
save_path = "/home/felix/rapid_storage_2/Broccoli/"

processing_params = Dict()
# Specify what data is laoded (or how)
processing_params["MultiExp"] = true # Flag for Multiple Exposures
# Preprocessing 
processing_params["NND"] = false # nothing for not doing list of params []
# Weight Matrix Specifix
processing_params["WeightPerColor"] = true # Use seperate weight per color
processing_params["ContrastWeightScaling"] = exp_scaling_contrast#exp_scaling_contrast # Broadcastable Function to apply to normalized contrast weights
processing_params["MFocus"] = [1,0,0] # Focus == [contrast_weigth, saturation_weight, exp_weight]
processing_params["MHDR"] = [0.5,1,0.6] # HDR == [contrast_weigth, saturation_weight, exp_weight]
# Pyramid Construction/Reconstruction Specific
processing_params["IgnorePyramidLevels"] = []  # List of pyramid Levels to be ignored
processing_params["BilateralFilterLvls"] = false # Not Worth using: List of levels on which to apply bilateral filter
# Location to save pre HDR pyramid and intermediate images (to check individual results)
processing_params["IntermediateSaveLocation"] = "$(save_path)MKR_"


# Actually used params

processing_params["prec32"] = true # Flag to do 32 bit rather than 64 bit weight creation
processing_params["ContrastFunction"] = "LoG" # Contrast Function to use STD or LoG --> LoG takes far longer (kernel is bigger 3x3 vs 7x7 but less often computed)
processing_params["Greyscaler"] = Grey_lstar # Greyscale : Only LoG contrast function
processing_params["blackpoint"] = [0,0,0]#trunc.(UInt16, [15,6,11] ./255 .* (2^16-1)) # 90,1,45

#FocusFusion_NND(image_directory, save_path, processing_params)
FocusFusion_Reduced_Memory(image_directory, save_path, processing_params)

println("Took $(time()-start)s")