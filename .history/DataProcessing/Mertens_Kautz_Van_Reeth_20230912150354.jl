


#=
This will contain the Mertens-Kautz-Van Reeth algorithm for HDR blending used by enfuse 
https://mericam.github.io/papers/exposure_fusion_reduced.pdf
This can be used for creating ldrs from a stack of different exposure images, and can be used
to do focus stacking in the cotrast weight is maximized, this will be used in our algorithm

The matlab code they provide is simply rewritten for julia here

Get image objects I, with size (w, h, color, px_val_for each iomage)

Run exposure fusion to get results with args (I, m) where m is : [contrast_weigth, saturation_weight, exp_weight]


The thing to execute here is 
exposure_fusion(Imagedat, m)






Exposure_fusion.m transcribed
Reconstruct_Laplacian_Pyramid.m transcribed
upsameple.m  transcribed
pyramid_filter transcribed
Laplacian_Pyramid.m transcribed
Gaussian_Pyramid.m transcribed
downsample.m transcribed but neglected symmetric boundary on downsamples imfilter so using repliate atm 



imfilter is not a function in julia, to solve this manually implemented border replicate function and imfilter 
But was too lazy too fill the entire border replicate so got missings in the corners which propagated as nans when the dtype changed

My conv uses fft and appends dim already, couldnt figure out what the border behavior is, but decided to neglect it and just use DFT, without adding padding, as padding isnt really required here, since we are using far less data thanis an image


=#
using Statistics

Pyramid_Filter = [.0625, .25, .375, .25, .0625] # TODO: We call transpose on this at some point we might need to explicitly declare this as a vector

MaximumPyramid = nothing # put to 5, check for nan before reconstructing and set to 0

Set_NAN_to_zero = false

function imfilter(mono, h, boundary)
    #=
    Naming and args from matlab only mono and h used
    mono -- greyscale image
    h -- kernel
    
    Implemented using DSP convolution, since we want replicate we make a new array with extra columns and rows depending on filter size
    to achieve the replicate feature, the central portion of the array is a view of the original to save space
    =#
    # Make larger array 
    if true # Temporary to check if kernel extend is the issue
        if boundary == "replicate"
            kernel_extend = trunc(Int,size(h)[1] / 2)
            extended = make_padding(mono, kernel_extend) 
        elseif boundary == "symmetric"
            throw("Not implemented")
        else
            throw("Not implemented")
        end

        return DSP.conv(extended, h)
    else
        return DSP.conv(mono, h)
    end
end


function Gaussian_Pyramid(I,nlev=nothing)
    size_ = size(I)
    r = size_[1]
    c = size_[2]
    if (isnothing(nlev) & isnothing(MaximumPyramid))
        nlev = floor(log(min(r,c)) / log(2))
    elseif !isnothing(MaximumPyramid)
        nlev = MaximumPyramid    
    end
    #pyr = Dict(1.0 => I)
    pyr = Dict()
    filter = Pyramid_Filter
    # Make copy, assigning now messes up some type stuff, really weird error
    J = I
    for l in 2.0:nlev
        I=downsample(I,filter)
        pyr[l] = I
        # Check for nan
        if any(isnan, I)
            println("$l in Gaussian Pyramid has nan")
            if Set_NAN_to_zero
                println("Setting nan to 0")
                pyr[l][isnan.(pyr[l])] .= 0
            end
        end
    end
    pyr[1.0] = J
    return pyr
end

function Laplacian_Pyramid(I,nlev=nothing)
    size_ = size(I)
    r = size_[1]
    c = size_[2]

    if (isnothing(nlev) & isnothing(MaximumPyramid))
        nlev = floor(log(min(r,c)) / log(2))
    elseif !isnothing(MaximumPyramid)
        nlev = MaximumPyramid    
    end
    J = convert(Array{Float64}, I)
    # Recursively build pyramid -- originally used cell, dict should suffice with matrix entries
    pyr = Dict()
    filter = Pyramid_Filter
    #if any(isnan, J) println("Laplacian Pyramid image after convert has none") end
    for l in (1:nlev-1)
        I = downsample(J, filter)
        #if any(isnan, I) println("Laplacian Pyramid image after downsample") end
        odd = 2 .*size(I) .- size(J)
        # Store difference between image and upsampled low pass version
        #if any(isnan, upsample(I,odd,filter)) println("Laplacian Pyramid image after upsample") end
        pyr[l] = J-upsample(I,odd,filter)
        J = I
        # Check for nan
        if any(isnan, pyr[l])
            println("$l in Laplacian Pyramid has nan")
            if Set_NAN_to_zero
                println("Setting nan to 0")
                pyr[l][isnan.(pyr[l])] .= 0
            end
        end
    end
    pyr[nlev] = J
    # Check for nan
    if any(isnan, pyr[nlev])
        println("$nlev in Laplacian Pyramid has nan")
        if Set_NAN_to_zero
            println("Setting nan to 0")
            pyr[nlev][isnan.(pyr[nlev])] .= 0
        end
    end
    return pyr
end

function downsample(I, filter)
    size_ = size(I)
    border_mode = "replicate" # FIXME: This is supposed to be symmetric but couldnt bother to implement
    if any(isnan, I) println("Image given to downsample has nan") end
    R = imfilter(I,filter,border_mode)    #horizontal
    if any(isnan, R) println("downsample first filter has nan") end
    R = imfilter(R,filter',border_mode)    #vertical
    if any(isnan, R) println("downsample second filter has nan") end
    # decimate
    r = size_[1]
    c = size_[2]
    R = R[1:2:r, 1:2:c, :] 
    return R 
end

function upsample(I, odd, filter)
    # Create padded aray
    extended = make_padding(I, 1)
    size_ = size(extended)
    r,c = size_[1:2].*2
    k = size_[3]
    R =zeros(r,c,k)
    if any(isnan, I) println("Image given to upsample has nan") end
    # In new aray padding is 2 px 
    R[1:2:r, 1:2:c, :] = extended .* 4
    R = imfilter(R, filter, "replicate") # horizontal
    if any(isnan, R) println("upsample first filter has nan") end
    R = imfilter(R, filter', "replicate") # vertical -> Prime does transpose TODO: CHeck this actually works
    if any(isnan, R) println("upsample second filter has nan") end
    # Remove border and return
    return R[3:r-2-odd[1] , 3:c-2-odd[2] , :]
end


function Reconstruct_Laplacian_Pyramid(pyr)
    nlev = length(pyr)
    # Start with low pass residual
    R = pyr[nlev]
    filter = Pyramid_Filter
    for l in ((nlev-1):-1:1)
        # Upsample, add to current level
        odd = 2 .*size(R) .- size(pyr[l])
        R = pyr[l] + upsample(R,odd,filter)
        if any(isnan, R)
            println("$l in Reconstruct_Laplacian_Pyramid has nan")
            if Set_NAN_to_zero
                println("Setting nan to 0")
                R[isnan.(R)] .= 0
            end
        end
    end
    return R
end


function Exposure_Fusion(I, m)
    dims = size(I) 
    r = dims[1] # Width 
    c = dims[2] # Height
    N = dims[4] # N images

    W = ones(r,c,N)# Fin mat #TODO: dtype?
    contrast_coeff = m[1]
    saturation_coeff = m[2]
    exposure_coeff = m[3]
    
    # Apply weights --- When this works add filter weight for distance from center of image for mosaiking
    if (contrast_coeff > 0)
        W = W.*contrast(I).^ contrast_coeff
    end

    if (saturation_coeff > 0)
        W = W.*saturation(I).^ saturation_coeff
    end

    if (exposure_coeff > 0)
        W = W.*well_exposedness(I).^ exposure_coeff
    end

    #Normalize  : Assure sum to 1 per pixel
    W .+= 1e-10 # Avoid 0 division
    # Here matlabs repmat is used the first input is taken and a new matrix is created with the first input placed in each dimension N times as specified by the second argument
    # Julia repeat will be used which also supports inner repitions so we specify outside
    W = W./repeat(sum(W,dims=3), outer=[1,1,N])
    println("Creating empty gaussian pyramid")
    # Create empty gaussian pyramid
    pyr = Gaussian_Pyramid(zeros(r,c,3))# TODO: dtype
    nlev = length(pyr)
    println("Do multires blending")
    # Multires blending
    for i in 1:N
        #Construct pyramid from each input
        # W is a matrix due to earlier broadcasting, so we do this to get an array, makes a copy # TODO: Make this a view and find a way to nest the least dimension more efficiently
        pyrW = Gaussian_Pyramid(W[:,:,i])#reshape([W[j,:,i] for j in 1:size(W, 1)], (size(W)[1:2]...,1))) 
        pyrI = Laplacian_Pyramid(I[:,:,:,i]) # TODO: This generates nans on the 7th run in l
        println("Blend $(i)")
        for l in 1:nlev
            w = repeat(pyrW[l], outer= [1,1,3])
            pyr[l] = pyr[l] + w.*pyrI[l]
            # Check for nan
            if any(isnan, pyr[l])
                println("$l in exposure fusion Pyramid has nan")
                if Set_NAN_to_zero
                    println("Setting nan to 0")
                    pyr[l][isnan.(pyr[l])] .= 0
                end
            end
        end
    end
    println("Reconstructing laplacian pyramid")
    return Reconstruct_Laplacian_Pyramid(pyr)
end


function contrast(I)
    h = [[0,1,0] [1, -4, 1] [0,0,1]] # Laplacian filter
    size_ = size(I)
    N = size_[4]
    C = zeros(size_[1], size_[2], N)
    for i in 1:N
        mono = rgb2array(I[:,:,:,i]) #  Average greysacale  Implemented as matlab copy
        C[:,:,i] = abs.(imfilter(mono,h,"replicate"))[3:end-2, 3:end-2] # Implemented as matlab copy 
    end
    return C
end


function make_padding(mono, kernel_extend)
    #=
    Adds padding on each side of size kernel_extend
    =#
    size_ = size(mono)
    if ndims(mono) == 3
        extended = Array{eltype(mono[1,1,1])}(undef, size_[1] + 2*kernel_extend, size_[2] + 2*kernel_extend, size_[3])
        extended .=0
        # Populate inner
        extended[kernel_extend+1:size_[1]+kernel_extend , kernel_extend+1:size_[2]+kernel_extend , :] = mono
        # Populate boundaries
        # Assign rows without corners
        for i in 1:kernel_extend extended[i, kernel_extend+1:end-kernel_extend,:] = mono[1, :,:] end 
        for i in size_[1]+kernel_extend:size(extended)[1] extended[i, kernel_extend+1:end-kernel_extend,:] = mono[end, :,:] end
        # Assign columns with corners
        for i in kernel_extend:-1:1 extended[:, i,:] .= extended[:, i+1,:] end
        for i in size_[2]+kernel_extend:size(extended)[2] extended[:, i,:] .= extended[:, i-1,:] end
    else
        extended = Array{eltype(mono[1,1])}(undef, size_[1] + 2*kernel_extend, size_[2] + 2*kernel_extend)
        extended .= 0
        # Populate inner
        extended[kernel_extend+1:size_[1]+kernel_extend , kernel_extend+1:size_[2]+kernel_extend] = mono
        # Populate boundaries 
        # Assign rows without corners
        for i in 1:kernel_extend extended[i, kernel_extend+1:end-kernel_extend] = @view mono[1, :] end 
        for i in size_[1]+kernel_extend:size(extended)[1] extended[i, kernel_extend+1:end-kernel_extend] = @view mono[end, :] end
        # Assign columns with corners
        for i in kernel_extend:-1:1 extended[:, i] .= @view extended[:, i+1] end
        for i in size_[2]+kernel_extend:size(extended)[2] extended[:, i] .= @view extended[:, i-1] end
    end
    # In case the above needs to be checked (couldnt figure out how to broadcast checkign for undef)
    #if any(x->x==0, extended) & !any(x->x==0, mono) println("Make padding still wrong")  end
    return extended
end

function rgb2array(I)
    #=
    Greyscale average color conversion 
    Indexing changes from w,h,1 to w,h
    =#
    conv_const = [0.212671, 0.715160, 0.072169]::Array{Float64}
    N = size(I, 1)::Int
    M = size(I, 2)::Int
    greyscale = Array{Float64}(undef, N,M)
    # We expect a 3 d image
    for i in 1:N
        for j in 1:M
            # Compute pixel wise Y :: TODO: Test if @view is more efficient here, based on docs it should be
            @inbounds greyscale[i,j] = sum(I[i,j,:].*conv_const)
            # Convert to luminance  (cond ? if true : if false)
            @inbounds greyscale[i,j] > 0.008856 ? greyscale[i,j] = 116*greyscale[i,j]^(1/3) - 16 :  greyscale[i,j] = 903.3* greyscale[i,j]
        end
    end 
    return greyscale
    #return mean(I, dims = 3)[:,:,1]
end

function saturation(I)
    #=
    Computed for each pixel as std between colors
    =# 
    size_ = size(I)
    N = size_[4]
    C = zeros(size_[1], size_[2], N)
    for i in 1:N
        # Sat computed as std of color channels
        R = @view I[:,:,1,i]
        G = @view I[:,:,2,i]
        B = @view I[:,:,3,i]
        μ = (R+G+B)/3
        C[:,:,i] = sqrt.(((R - μ).^2 + (G - μ).^2 + (B - μ).^2)/3)
    end
    return C
end

function well_exposedness(I)
    size_ = size(I)
    N = size_[4]
    sig = 0.2
    C = zeros(size_[1], size_[2], N)
    for i in 1:N
        # In matlab implemented as:
        #R = exp(-0.5*(I[:,:,1,i] .- 0.5).^2/sig.^2);
        # as this is the gaussian we implement as 
        R = exp.(-0.5*((I[:,:,1,i] .- 0.5).^2)/(sig^2));
        G = exp.(-0.5*((I[:,:,2,i] .- 0.5).^2)/(sig^2));
        B = exp.(-0.5*((I[:,:,3,i] .- 0.5).^2)/(sig^2));
        C[:,:,i] = R.*G.*B
    end
    return C
end