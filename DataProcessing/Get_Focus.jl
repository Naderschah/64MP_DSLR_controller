#=
Here all code regarding focus stacking and HDR will live


Enfuse uses : Mertens-Kautz-Van Reeth exposure fusion algorithm for variable focus stacking
Look into for HDR




Functions:
ComputeFocusMap -> Computes an images focus map (greyscale laplacian) arguments are image and grey_project (currently l-star and average)
Rescale -> Scales input image from Int16 to Float64
Laplacian -> Computes laplacian image by convolution
Grey_Project_*:
    Average -> Average value of rgb
    l_star -> Computes LAB color space L value
    individual -> reshapes array so that Laplacian transform is run on each color should chromatic abberations become large :: To be implemented
=#

using StatsBase
using Match
using LibRaw
using JLD2 # Saving data
using BenchmarkTools
using DSP
using PyCall

function memuse()
    #=
    Returns memory use from top 
    =#
    perc = [i for i in eachmatch(r"(\w+)",split(read(`top -bn1 -p $(getpid())`, String), "\n")[end-1])][end-5:end-4]
    return parse(Float64, perc[1].match*"."*perc[2].match)
end

function GetImageBPS(path)
    #=
    Libraw only returns a pointer to the bps for some reason so we use exiftools to load the bps
    =# 
    res = read(`exiftool $path`,String)
    return parse(Int,split([i for i in split(res, "\n") if occursin("Bits", i)][1], ":")[2])
end

function GetImageData(path, bps, debayer="No")
    #=
    Uses libraw(-dev) c raw data loader and then postprocess the image to be usable 
    debayering can be skipped through the No parameter so that the image is returned as half the width and height
    =#
    im_ = LibRaw.RawImage(path)
    # This gives us the data order
    if debayer != "No"
        colors = LibRaw.color_description(im_)
        im = LibRaw.demoisaic(LibRaw.BayerAverage(), im_)
        green_index = findall("G", colors)
        unpack!(im_)
        # Make first halfed
        im[:,:,green_index[1][1]] = im[:,:,green_index[1][1]]./2 + im[:,:,green_index[2][1]]./2
        #Normalize to 0-1
        ret = im[:,:,1:3]./(2^bps-1)
        LibRaw.close(im_)
        return ret
    else
        # Unpack hope it fixes memory leak
        unpack!(im_)
        im = NoDebayer(im_,bps) 
        LibRaw.close(im_)
        return im
    end
end

function GetImageDataPy(path, bps, debayer="No", save="")
    # If save is given a string it iwll save it using the jld2 format
    # Use python rawpy as LibRaw causes mamory leak
    rawpy = pyimport("rawpy") 
    im_1 = rawpy.imread(path)
    im_ = im_1.raw_image_visible
    im_1.close()
    h,w = size(im_)

    w = Int(w/2)
    h = Int(h/2)
    ret = zeros(h,w,3)
    # Create final array
    @inbounds ret[:,:,1] .= im_[2:2:end,2:2:end]./(2^bps-1) #(reshape(im[color_index .== 1]./(2^bps-1), (w,h,1)))
    @inbounds ret[:,:,3] .= im_[1:2:end,1:2:end]./(2^bps-1) #(reshape(im[color_index .== 3]./(2^bps-1), (w,h,1)))
    @inbounds ret[:,:,2] .= (im_[1:2:end,2:2:end]./2 + im_[2:2:end,1:2:end]./2)./(2^bps-1) #reshape((im[color_index .== 2] ./2 + im[color_index .== 4]./2)./(2^bps-1), (w,h,1))
    if save !=""
        save_object(save, ret)
    end
    return ret
end

function NoDebayer(im,bps)
    #=
    Takes colors and makes rgb image by simply extracting each color rather than interpolating 
    =#
    # Scale - first check image actually has data
    if sum(size(LibRaw.raw_image(im)) .> 0) == 2
        im_ = copy(LibRaw.raw_image(im)./(2^bps-1))
    else 
        println("Image has no data")
        return [0]
    end
    ret = zeros(Int(LibRaw.raw_height(im)/2),Int(LibRaw.raw_width(im)/2),3)
    # Avg green
    @inbounds ret[:,:,2] .= im_[1:2:end,2:2:end]./2 + im_[2:2:end,1:2:end]./2
    # Create array reference
    @inbounds ret[:,:,1] .= im_[2:2:end,2:2:end] #(reshape(im[color_index .== 1]./(2^bps-1), (w,h,1)))
    @inbounds ret[:,:,3] .= im_[1:2:end,1:2:end] #(reshape(im[color_index .== 3]./(2^bps-1), (w,h,1)))
    return ret
end



function ComputeFocusMap(image, grey_project)
    # Rescaling didnt end up being required for the bitmap format
    #image = Rescale(image)
    grey(image) = @match grey_project begin
        "l-star" => Grey_Project_l_star(image)
        "average" => Grey_Project_Average(image)
        "individual" => throw("individual laplacian transform to be implemented")
        _ => throw("Projection $grey_project not implemented")
    end
    return @fastmath LoG(grey(image))
end

function MakeFocusedImage(work_dir,im_path, y, z, bps) #map_dir, im_save_path,depth_save_path, raw_dir
    #=
    Loads focus maps and finds depthmap, then uses raw_dir to make stacked image
    =#
    # Paths
    t="_"
    map_dir = "$work_dir/focus/$z/$y/"
    im_save_path = "$work_dir/focus_stacked/$y$t$z.jld2" # TODO FIXME when works
    depth_save_path = "$work_dir/depthmaps/$y$t$z.jld2"
    depth_save_path_conf = "$work_dir/depthmaps/$y$t$z$t"*"confidence.jld2"
    raw_alter = "$work_dir/raw_j/"
    raw_dir = "$work_dir/raw/"

    map_ = nothing
    image = nothing
    map_loc = nothing
    for i in readdir(map_dir,join=true)
        tmp = abs.(load_object(i)[2:end-1,2:end-1])
        if isnothing(map_)
            map_ = fill(0.1, size(tmp)) # Fill with 0.10 as anything below is not worth considering
            map_loc = zeros(size(tmp))
        else
            bool_array = (tmp .> map_)
            # If any are true we also load the image
            if any(bool_array)
                println("Replacing for point $i")
                x = split(last(split(i,"/")),".")[1]
                # Save x position in map dist array
                map_loc[bool_array] .= parse(Int, x)
                # Create map confidence entry
                map_[bool_array] .= tmp[bool_array]
                # We laod the image
                if split(i, ".")[2] == "dng"
                    img = GetImageDataPy("$im_path/$x$t$y$t$z$t"*"NoIR.dng", bps)
                else
                    img = load_object("$im_path/$x$t$y$t$z$t"*"NoIR.jld2")
                end
                # Create focus image if doesnt exist with same shape and dtype
                if isnothing(image)
                    image = similar(img)
                end
                # Overwrite image parts
                image[bool_array, :] = img[bool_array, :]
            end
        end
        tmp = nothing
    end
    save_object(depth_save_path, map_loc)
    save_object(depth_save_path_conf, map_)
    # Save TODO check if tiff works
    save_object(im_save_path, image)
    GC.gc()
end


function Laplacian(image)
    #=
    We compute the laplacian, to do so we convolve with a 3x3 kernel with center weight -4 and all direct edges weight 1
    =#
   return DSP.conv([[0,1,0] [1,-4,1] [0,1,0]], image)
end

function LoG(image)
    #=
    Laplacian of Gaussian 1.4\sigma --> less sensitive to noise 
    =#
    LoG_kernel = [[0,1,1,2,2,2,1,1,0] [1,2,4,5,5,5,4,2,1] [1,4,5,3,0,3,5,4,1] [2,5,3,-12,-24,-12,3,5,2] [2,5,0,-24,-40,-24,0,5,2] [2,5,3,-12,-24,-12,3,5,2] [1,4,5,3,0,3,5,4,1] [1,2,4,5,5,5,4,2,1] [0,1,1,2,2,2,1,1,0] ]
    # We index so that the 1 overshoot in each direction remains consistent
    return DSP.conv([[0,1,0] [1,-4,1] [0,1,0]], image)[5:end-4,5:end-4]
end

function Grey_Project_Average(image)
    #=
    Grey scale projection average last axis
    =#
    N = size(image, 1)
    M = size(image, 2)
    K = size(image, 3)
    greyscale = Array{Float64}(undef, N, M)
    for i in 1:N
        for j in 1:M
            for k in 1:K
                @inbounds greyscale[i,j] += image[i,j,k]/3
            end
        end
    end
    return greyscale
end

function Grey_Project_l_star(image)
    #=
    The L* refers to the LAB (CIELAB) color spaces L* channel so we jsut transform the color space
    We only compute L* as we dont care about a and b
    
    Y = 0.212671*R + 0.715160*G + 0.072169 * B

    if Y > 0.008856 : L = 116*Y^(1/3) -16
    else: L = 903.3 * Y     
    =#
    conv_const = [0.212671, 0.715160, 0.072169]::Array{Float64}
    N = size(image, 1)::Int
    M = size(image, 2)::Int
    greyscale = Array{Float64}(undef, N,M)
    # We expect a 3 d image
    for i in 1:N
        for j in 1:M
            # Compute pixel wise Y :: TODO: Test if @view is more efficient here, based on docs it should be
            @inbounds greyscale[i,j] = sum(image[i,j,:].*conv_const)
            # Convert to luminance  (cond ? if true : if false)
            @inbounds greyscale[i,j] > 0.008856 ? greyscale[i,j] = 116*greyscale[i,j]^(1/3) - 16 :  greyscale[i,j] = 903.3* greyscale[i,j]
        end
    end
    return greyscale
end
    



