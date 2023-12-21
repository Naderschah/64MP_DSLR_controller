


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



To fix singleton:
We have tst with size : (40, 30, 2, 3, 10)
size(tst[:,:,1:1,:,:]) == (40, 30, 1, 3, 10)
size(tst[:,:,1,:,:]) == (40, 30, 3, 10)

Exposure_fusion.m transcribed
Reconstruct_Laplacian_Pyramid.m transcribed
upsameple.m  transcribed
pyramid_filter transcribed
Laplacian_Pyramid.m transcribed
Gaussian_Pyramid.m transcribed
downsample.m transcribed but neglected symmetric boundary on downsamples imfilter so using repliate atm 


Indexing Might still be wrong (one to bottom one to right)


NearestNeighborDeconvolution doesnt change anything --> also tried weighing it with the contrast weights, but no usefull result -->  


=#



using Statistics
using JLD2
using Images
using Dates

Pyramid_Filter = [.0625, .25, .375, .25, .0625] # TODO: We call transpose on this at some point we might need to explicitly declare this as a vector
LoG_kernel = [[0,1,1,2,2,2,1,1,0] [1,2,4,5,5,5,4,2,1] [1,4,5,3,0,3,5,4,1] [2,5,3,-12,-24,-12,3,5,2] [2,5,0,-24,-40,-24,0,5,2] [2,5,3,-12,-24,-12,3,5,2] [1,4,5,3,0,3,5,4,1] [1,2,4,5,5,5,4,2,1] [0,1,1,2,2,2,1,1,0] ]



function Exposure_Fusion(I, processing_params)
    #=
    Does Mertens Kautz Van Reeth algorithm,
    Extended by adding Bilinear filter on the [7,8,9] th levels of the generated pyramid (assumes 4k images)
        Only on these as it is computationally expensive to run on all, and these appeared to be the noisiest
    Prior to MKvR the Nearest Neighbor Deconvolution is applied to remove some of the out of focus bluring
        Different implementations of this idea are to follow

    I -> Image stack with 4 dims : 4056,3040,E,3,N where E is teh number of exposures and N is the number of images to process
        ---> E can be omitted if MultiExp is false
    m -> Coefficients for contrast,saturation,exposure respectively
    ignore_level -> In case any laplacian pyramid levels should be ignored
    =#
    stp_count = 1
    # Check that flags are compatible
    if (processing_params["ContrastFunction"] == "STD") & !processing_params["WeightPerColor"]
        throw("If standard deviation Contrast function is used weight per color must be on")
    end

    # Replace string flags
    if processing_params["Greyscaler"] == "l-star"
        processing_params["Greyscaler"] = Grey_lstar
    elseif processing_params["Greyscaler"] == "Mean"
        processing_params["Greyscaler"] = Grey_Mean
    end

    printstyled("$(stp_count): Normalizing Images\n", color = :green, bold = true)
    stp_count+=1
    if eltype(I) ∈ (UInt, UInt128, UInt16, UInt32, UInt64, UInt8)
        I = I ./ typemax(eltype(I))
    else
        throw("Data should be supplied as UInt dtype, if not possible add handling for normalization here")
    end

    printstyled("$(stp_count): Computing Weight Matrix\n", color = :green, bold = true)
    stp_count+=1
    println("Current Time $(Dates.format(now(), "HH:MM:SS"))")
    size_ = size(I) 
    r = size_[1] # Width 
    c = size_[2] # Height
    N = size_[end] # N images
    if processing_params["MultiExp"] e = size_[3] # Exposures 
    end

    # Compute weight matrix based on contrast exposure saturation 
    W = Compute_Weight_Matrix(I,processing_params,processing_params["MFocus"])
    println("Finished Weight Matrix")
    println("Current Time $(Dates.format(now(), "HH:MM:SS"))")

    # 5 im dim loop -> with exposures seperated 
    if processing_params["MultiExp"]
        # Using dict cause why not -> Stores each pyramid
        printstyled("$(stp_count): Focus Stacking Each exp\n", color = :green, bold = true)
        stp_count+=1
        out_dict = Dict()
        println("Current Time $(Dates.format(now(), "HH:MM:SS"))")
        for i in 1:e # Iterate each exposure stack
            printstyled("Starting focus stack of exposure $i/$e\n", color = :blue)
            out_dict[i] = DoFusion(remove_singleton(I[:,:,i,:,:]), remove_singleton(W[:,:,i,:,:]), r, c,N, processing_params)
        end
        println("Finished Focus Stacking")
        println("Current Time $(Dates.format(now(), "HH:MM:SS"))")
        printstyled("$(stp_count): Saving jld2 pyramid and recreated Images\n", color = :green, bold = true)
        stp_count+=1
        interm = processing_params["IntermediateSaveLocation"]
        JLD2.save_object("$(interm)Pyramid.jld2", out_dict)
        println("Current Time $(Dates.format(now(), "HH:MM:SS"))")
        dims = size(I) 
        r, c, N = dims[1:3] # Now we are only left with the individual exposures (index 3 == number exposures)
        # Dont need the original image array anymore, so overwrite for garbage collect:
        I = Array{Float64}(undef, r, c, 3, N)
        # Now that we cleared up ram reconstruct the images and then do HDR
        for i in 1:N
            I[:,:,:,i] = Reconstruct_Laplacian_Pyramid(out_dict[i])
            debug_save(I[:,:,:,i], "$(interm)_$i.png")
        end
        println("Reconstructed Images")
        println("Current Time $(Dates.format(now(), "HH:MM:SS"))")
        out_dict = nothing
        printstyled("$(stp_count): Computing HDR weight matrix\n", color = :green, bold = true)
        stp_count+=1
        processing_params["MultiExp"] = false
        processing_params["ContrastFunction"] = "LoG"
        W =Compute_Weight_Matrix(I,processing_params,processing_params["MHDR"])
        printstyled("$(stp_count): Doing Exposure Fusion and Reconstruction\n", color = :green, bold = true)
        stp_count+=1
        
        return Reconstruct_Laplacian_Pyramid(DoFusion(I, W, r, c, N, processing_params))
    # 4 im dim loop -> without exposures seperated
    else
        printstyled("$(stp_count): Focus Stacking\n", color = :green, bold = true)
        stp_count+=1
        return Reconstruct_Laplacian_Pyramid(DoFusion(I, W, r, c,N, processing_params))
    end
end


function DoFusion(I, W, r, c,N, processing_params)
    #=
    Actually fuses after preprocessing
    =#
    println("Creating empty gaussian pyramid")
    pyr = Gaussian_Pyramid(zeros(r,c,3))
    nlev = length(pyr)
    I = remove_singleton(I)
    println("Do multires blending")
    count = 1
    for i in 1:N
        #Construct pyramid from each input image
        if processing_params["WeightPerColor"] pyrW = Gaussian_Pyramid(remove_singleton(W[:,:,:,i])) 
        else pyrW = Gaussian_Pyramid(remove_singleton(W[:,:,i]))
        end

        pyrI = Laplacian_Pyramid(remove_singleton(I[:,:,:,i]))
        println("Blend $(i)")
        # Iterate over the individual levels and add the weighted level from each input
        for l in 1:nlev
            if l ∉ processing_params["IgnorePyramidLevels"]
                # Fix element shape 
                if processing_params["WeightPerColor"] w = pyrW[l]
                else w = repeat(pyrW[l], outer= [1,1,3])
                end
                # Take bilateral filter on subset of pyramid levels
                if l ∈ processing_params["BilateralFilterLvls"] 
                    # The proper implementation does not work, still needs fixing
                    #recreated = LapLevelDenoising(pyrI[l-1], pyrI[l], processing_params["Greyscaler"], 0.5)
                    no_grey(x) = x # Dummy grey project
                    for k in 1:3 pyr[l][:,:,k] = BilaterFilter(pyrI[l][:,:,k],1.4, NoiseEstimate(pyrI[l]),no_grey) end
                end
                pyr[l] = pyr[l] + w.*pyrI[l] #
                count +=  1
            end
        end
    end
    return pyr
end


function imfilter(mono, h, boundary="replicate")
    #=
    Naming and args from matlab only mono and h used
    mono -- greyscale image
    h -- kernel
    
    Implemented using DSP convolution, since we want replicate we make a new array with extra columns and rows depending on filter size
    to achieve the replicate feature, the central portion of the array is a view of the original to save space
    =#
    # Make larger array 
    if boundary == "replicate"
        kernel_extend = trunc(Int,size(h)[1] / 2)
        extended = make_padding(mono, kernel_extend) 
    elseif boundary == "symmetric"
        throw("Not implemented")
    else
        throw("Not implemented")
    end

    return DSP.conv(extended, h)
end


function Gaussian_Pyramid(I,nlev=nothing)
    #=
    Construct the gaussian Pyramid, nlev determines maximum level --> Automatically computed based on maximum posible
    =#
    size_ = size(I)
    r = size_[1]
    c = size_[2]
    if (isnothing(nlev))
        nlev = floor(log(min(r,c)) / log(2))
    end
    pyr = Dict()
    filter = Pyramid_Filter
    # Make copy, assigning now messes up some type stuff, really weird error
    J = I
    for l in 2.0:nlev
        I=downsample(I,filter)
        pyr[l] = I
    end
    pyr[1.0] = J
    return pyr
end

function Laplacian_Pyramid(I,nlev=nothing)
    size_ = size(I)
    r = size_[1]
    c = size_[2]

    if (isnothing(nlev))
        nlev = floor(log(min(r,c)) / log(2))
    end
    J = convert(Array{Float64}, I)
    # Recursively build pyramid
    pyr = Dict()
    filter = Pyramid_Filter
    for l in (1:nlev-1)
        I = downsample(J, filter)
        if any(isnan.(I)) println("Downsample in laplacian pyramid level $l has nan") end
        odd = 2 .*size(I) .- size(J)
        # Store difference between image and upsampled low pass version
        pyr[l] = J-upsample(I, odd, filter)
        if any(isnan.(upsample(I, odd, filter))) println("Upsample in laplacian pyramid level $l has nan")end
        J = I
    end
    pyr[nlev] = J
    return pyr
end

function downsample(I, filter)
    # Apply paded filter
    if ndims(filter) == 1  filter=filter*filter' end
    extended = imfilter(I,filter,"replicate")
    #extended = imfilter(extended,filter,"replicate")
    # remove padding 
    fe = trunc(Int,size(filter)[1]/2)
    multip = 2 # 2 for each imfilter
    if ndims(I) == 3
        extended=extended[multip*fe+1:end-multip*fe, multip*fe+1:end-multip*fe,:]
        # return every other element
        return extended[1:2:end,1:2:end,:]
    elseif ndims(I) == 4
        extended=extended[multip*fe+1:end-multip*fe, multip*fe+1:end-multip*fe,:,:]
        # return every other element
        return extended[1:2:end,1:2:end,:,:]
    else
        throw("Downsample not defined for $(ndims(I))")
    end
end

function upsample(I,odd, filter)
    #=
    Odd --> if the next image needs to be odd

    In downsample (if odd) the last element is dropped
    So in upsample we want to remove one element from the end (as it will be one dim larger than we want)

    If i always index as [1:2:r, 1:2:c, :] the image gets blacker in the top right spreading throughout the image
    if i always index as [2:2:r, 2:2:c, :] it gets whiter throughtout the image
        The above effects are made stronger by multipliers 

    If i assign half the value to both I get (or any variation of this) the image gets grainy --> doesnt focus either

    If i alternate the assignment I get the worst of both --> WIth 2* works pretty well

    Assigning first to one index running the imfilter then assigning to the second also appears to work pretty well

    Doing first the vertical then the horizontal kernel is equiv to running 2d seperable : when runnign half size same thing
    ---- doing the above but adding 2* on the second indexing after first filter 

    =#
    # Make zero array
    r,c = size(I)[1:2].*2
    if ndims(I) == 3
        R =zeros(r,c,3)
    else
        R =zeros(r,c)
    end
    # Get filter array expansion
    # Populate array, for the first we dont need to account for odd
    if !isnothing(filter)
        fe = trunc(Int,size(filter)[1]/2)
        R[1:2:r, 1:2:c, :] = I .* 4
        ## Fix borders in case we need odd sized array
        if odd[1] == 1 R = R[1+odd[1]:end,1:end,:] end
        if odd[2] == 1 R = R[1:end,1+odd[2]:end,:] end
        if ndims(filter) == 1 filter = filter*filter' end
        R = imfilter(R,filter,"replicate") 
        R = imfilter(R,filter,"replicate") 
        multip = 4 # 2 for each imfilter
        R = R[multip*fe+1:end-multip*fe, multip*fe+1:end-multip*fe,:]
    else
        if ndims(I) == 3
            R[1:2:r, 1:2:c, :] = I
            R[2:2:r, 2:2:c, :] = I
            ## Fix borders in case we need odd sized array
            if odd[1] == 1 R = R[1+odd[1]:end,1:end,:] end
            if odd[2] == 1 R = R[1:end,1+odd[2]:end,:] end
        else
            R[1:2:r, 1:2:c] = I
            R[2:2:r, 2:2:c] = I
            ## Fix borders in case we need odd sized array
            if odd[1] == 1 R = R[1+odd[1]:end,1:end] end
            if odd[2] == 1 R = R[1:end,1+odd[2]:end] end
        end
        
    end
    return R
    
end

function GaussianKernel(n, sigma)
    # Create symmetric gaussian kernel of size nxn
    mid = trunc(Int, (n)/2 )
    D1kern = [exp(-(i - mid)^2/(2*sigma^2)) for i in 1:n]
    D1kern ./= sum(D1kern)
    return D1kern*D1kern'
end

function Gaussian(xsquared, sigma)
    #=
    To create the gaussian kernel for Bilateral Filter
    =#
    return exp.(-0.5*(xsquared)/sigma^2)
end

function MakeEdgeMap(grey_project)
    #= 
    Function to be called after Bilateral filter to generate the edgemap for the laplacian sublevel
    =#
    # Do grey projection, LoG filter, scaling and indexing of extra dims
    extended = imfilter(grey_project,LoG_kernel,"replicate") 
    extended = normalize(extended, 0)
    multip = 2 # 2 for each imfilter
    fe = trunc(Int,size(LoG_kernel)[1]/2)
    # Remove bounds and make absolute and return
    return abs.(extended[multip*fe+1:end-multip*fe, multip*fe+1:end-multip*fe])
end

function Reconstruct_Laplacian_Pyramid(pyr)
    #=
    Takes pyramid and makes image
    =#
    println("Reconstructing Laplacian Pyramid")
    nlev = length(pyr)
    # Start with low pass residual
    R = pyr[nlev]
    filter = Pyramid_Filter
    for l in ((nlev-1):-1:1)
        # Upsample, add to current level
        odd = 2 .*size(R) .- size(pyr[l])
        R = pyr[l] + upsample(R,odd,filter)
        if any(isnan.(R)) println("Laplacian reconst at level $l has $(sum(isnan.(R))) Nan values") end
        
    end
    return R
end


function normalize(dat, offset)
    min_ = minimum(dat)
    max_ = maximum(dat)
    return (dat .-min_)./(max_.-min_ .- offset ) 
end

function Compute_Weight_Matrix(I,processing_params,m)
    #=
    Computes Weight matrix
    =#
    # Alias for readability
    prec32 = processing_params["prec32"] 
    # If lower precision for Weight matrix
    if prec32 I = float32.(I) end
    # Get dimensions
    size_ = size(I) 
    r = size_[1] # Width 
    c = size_[2] # Height
    N = size_[end] # N images
    if processing_params["MultiExp"] e = size_[3] # Exposures 
    end
    # Get shape for Weight matrix (colors and exposure dimension?)
    if processing_params["MultiExp"] W_shape = (processing_params["WeightPerColor"] ? (r,c,e,3,N) : (r,c,e,N))
    elseif !processing_params["MultiExp"] W_shape = (processing_params["WeightPerColor"] ? (r,c,3,N) : (r,c,N))
    end
    # Allocate weigth matrix
    W = (prec32 ? ones(Float32, W_shape) : ones(Float64, W_shape))
    # Extract selection coefficients
    contrast_coeff = m[1]
    saturation_coeff = m[2]
    exposure_coeff = m[3]
    # Apply weights
    if (contrast_coeff > 0)
        if processing_params["ContrastFunction"] == "STD"  
            # FIXME: STD is kind of broken for some reason the last exposure (or if there is only one that one) has only a subset of the exposed pixels, so we compute all and do max pool
            if processing_params["MultiExp"] 
                # Do processing on all
                if any(isnan.(I)) println("Nans in image $(sum(isnan.(I)))") end
                cont = (prec32 ? contrast_std(float32.(I)) : contrast_std(I))
                if any(isnan.(cont)) println("Nans in cont $(sum(isnan.(cont)))") end
                # Normalize each exposure contrast to itself as higher exp will have larger cont values
                for i in 1:e   cont[:,:,i,:,:] ./= maximum(cont[:,:,i,:,:])   end
                # Select maximum vals along exposure dimension and replicate to correct shape
                # This is so that all have the same in focus points
                cont = repeat(maximum(cont, dims=3), outer=[1,1,e,1,1])
                if any(isnan.(cont)) println("Nans in cont") end
            else
                throw("STD is broken and will not return the full in focus area, for single exposures LoG is recommended")
            end
        elseif processing_params["ContrastFunction"] == "LoG" cont = (prec32 ? contrast_LoG(float32.(I), processing_params["Greyscaler"], processing_params["WeightPerColor"]) : contrast_LoG(I, processing_params["Greyscaler"], processing_params["WeightPerColor"]))
        else throw("Contrast Function: $(processing_params["ContrastFunction"]) not recognized")
        end 
        # Here we maxpool cause something about focus STD is broken
        # Normalize 0 to 1
        max_ = maximum(cont)
        cont = cont ./ maximum(cont)
        if any(isnan.(cont)) println("Nans in cont Max val: $(max_)") end
        # contrast weight scaling
        if !isnothing(processing_params["ContrastWeightScaling"]) processing_params["ContrastWeightScaling"].(cont) end
        W = W.*cont.^ contrast_coeff

        # Compute Nearest Neighbor Deconvolution - Here to have contrast be an extra factor for it -- It seems to remove a similar value everywhere so we add a filter
        if processing_params["NND"]
            #printstyled("$(stp_count): Doing NND\n", color=:green, bold = true)
            printstyled(" Doing NND\n", color=:green, bold = true)
            #stp_count+=1
            # For each exposure
            if processing_params["MultiExp"]
                for i in 1:size(I)[3] I[:,:,i,:,:] = NearestNeighborDeconvolution(I[:,:,i,:,:], W[:,:,i,:,:], 0.4, nothing) end
            else
                I = NearestNeighborDeconvolution(I, 0.4, nothing)
            end
            if any(isnan.(I)) println("Images after NND have $(sum(isnan.(I))) Nan values") end
        end
    end
    if (saturation_coeff > 0) # Does not return per color
        sat = (prec32 ? saturation(float32.(I)) : saturation(I))
        sat = sat ./ maximum(sat)
        if ((ndims(sat) == (ndims(W)-1)) & (ndims(W) == 4)) # Occurs due to singleton removal on HDR -- idk how to fix this properly
            size_ = size(sat)
            sat = reshape(sat, size_[1], size_[2], 1, size_[3])
        end
        if processing_params["MultiExp"] sat = repeat(sat, outer=[1,1,1,3,1]) # Dims [im_y,im_z, exp, 3,N]
        elseif processing_params["WeightPerColor"] sat = repeat(sat, outer=[1,1,3,1])# Dims [im_y,im_z, 3,N]
        else sat = sat
        end
        W = W.*sat.^ saturation_coeff
    end
    if (exposure_coeff > 0) # Does not return per color
        exp = (prec32 ? well_exposedness(float32.(I)) : well_exposedness(I))
        exp =  exp ./ maximum(exp)
        # Occurs due to singleton removal on HDR -- idk how to fix this properly
        if ((ndims(exp) == (ndims(W)-1)) & (ndims(W) == 4)) 
            size_ = size(exp)
            exp = reshape(exp, size_[1], size_[2], 1, size_[3])
        end
        # Reshape so the sizes are the same
        if processing_params["MultiExp"] exp = repeat(exp, outer=[1,1,1,3,1]) # Dims [im_y,im_z, exp, 3,N]
        elseif processing_params["WeightPerColor"] exp = repeat(exp, outer=[1,1,3,1])# Dims [im_y,im_z, 3,N]
        else exp = exp
        end
        W = W.*exp.^ exposure_coeff
    end
    W .+= 1e-12 # Avoid 0 division
    #Normalize  : Assure sum to 1 per pixel
    if ndims(I)==5  W = W./repeat(sum(W,dims=5), outer=[1,1,1,1,N])
    elseif ndims(I)==4  W = W./repeat(sum(W,dims=4), outer=[1,1,1,N])
    else  W = W./repeat(sum(W,dims=3), outer=[1,1,N]) # Dims [im_y, im_z, N]
    end
    return W
end


function remove_singleton(im)
    #=
    Indexing sometimes leaves a singleton, TODO : Fix when understand why it leaves it sometimes but not always
    =#
    dim = 0
    while length(findall(size(im).==1)) > 0 # If more error will come later
        dim=findall(size(im).==1)[1]
        im = dropdims(im,dims=dim)
    end
    return im
end


function debug_save(tst, path)
    #=
    Saves image with only central portion considered for min max values --> Used when image colors get a bit fucky
    =#
    # Debugging save -- in case broken stacking computes max min from center of image
    min_ = minimum(tst[1000:end-1001, 1000:end-1001, :])
    max_ = maximum(tst[1000:end-1001, 1000:end-1001, :])
    tst  = (tst .-min_)./(max_.-min_ ) .* (2^16-1)
    tst[tst .< 0 ] .= 0
    tst[tst .> (2^16-1) ] .= 2^16-1
    Images.save(path, trunc.(UInt16,tst))
end

function make_padding(mono, kernel_extend)
    #=
    Adds padding on each side of size kernel_extend
    Defined for 2D and 3D arrays (im_dim_1, im_dim_2, [colors]) 
    =#
    # Remove dimensions of size 1 - why the fuck is this so annoying to do
    dim = 0
    if length(findall(size(mono).==1)) > 0 # If more error will come later
        dim=findall(size(mono).==1)[1]
        mono = dropdims(mono,dims=dim)
    end
    size_ = size(mono)
    if ndims(mono) == 3
        extended = Array{eltype(mono[1,1,1])}(undef, size_[1] + 2*kernel_extend, size_[2] + 2*kernel_extend, size_[3])
        #extended .=0
        # Populate inner
        extended[kernel_extend+1:size_[1]+kernel_extend , kernel_extend+1:size_[2]+kernel_extend , :] = mono
        # Populate boundaries
        # Assign rows without corners
        for i in 1:kernel_extend extended[i, kernel_extend+1:end-kernel_extend,:] = mono[1, :,:] end 
        for i in size_[1]+kernel_extend:size(extended)[1] extended[i, kernel_extend+1:end-kernel_extend,:] = mono[end, :,:] end
        # Assign columns with corners
        for i in kernel_extend:-1:1 extended[:, i,:] .= extended[:, i+1,:] end
        for i in size_[2]+kernel_extend+1:size(extended)[2] extended[:, i,:] .= extended[:, i-1,:] end
    elseif ndims(mono) == 2
        extended = Array{eltype(mono[1,1])}(undef, size_[1] + 2*kernel_extend, size_[2] + 2*kernel_extend)
        #extended .= 0
        # Populate inner
        extended[kernel_extend+1:size_[1]+kernel_extend , kernel_extend+1:size_[2]+kernel_extend] = mono
        # Populate boundaries 
        # Assign rows without corners
        for i in 1:kernel_extend extended[i, kernel_extend+1:end-kernel_extend] = mono[1, :] end 
        for i in size_[1]+kernel_extend:size(extended)[1] extended[i, kernel_extend+1:end-kernel_extend] = mono[end, :] end
        # Assign columns with corners
        for i in kernel_extend:-1:1 extended[:, i] .= extended[:, i+1] end
        for i in size_[2]+kernel_extend+1:size(extended)[2] extended[:, i] .= extended[:, i-1] end
    else
        throw("Padding Shouldnt be required for higher than 3D")
    end
    
    return extended
end

function contrast_LoG(I, grey_project=nothing, per_color=false)
    #=
    Contrast using Lagrangian of Gaussian
    I -> Image
    grey -> greyscale projector (if requiredd)

    TODO: How do i do this indexing without all the goddamn if statements - for all weight functions
    =#
    h = [[0,1,0] [1, -4, 1] [0,1,0]] # Laplacian filter
    # Lagrangian of Gaussian 1.4σ
    LoG_kernel = [[0,1,1,2,2,2,1,1,0] [1,2,4,5,5,5,4,2,1] [1,4,5,3,0,3,5,4,1] [2,5,3,-12,-24,-12,3,5,2] [2,5,0,-24,-40,-24,0,5,2] [2,5,3,-12,-24,-12,3,5,2] [1,4,5,3,0,3,5,4,1] [1,2,4,5,5,5,4,2,1] [0,1,1,2,2,2,1,1,0] ]
    # TODO : Try normalized log kernel here
    kernel_size = trunc(Int,size(LoG_kernel)[1])
    size_ = size(I)
    N = size_[end]
    # Check how output dimensions need to be, possible are (r,c,e,3,N) (r,c,e,N) (r,c,3,N) (r,c,N)

    if ((ndims(I) == 4) & (size_[3] == 3)) # Dims 4 corresponds to [im_y, im_z, colors, N_im] as no mono image should appear the size_[3] checking is technically not required 
        if per_color
            C = zeros(size_[1], size_[2], 3, N)
            for i in 1:N for j in 1:3  C[:,:,j,i] = abs.(DSP.conv(LoG_kernel, make_padding(I[:,:,j,i] , kernel_size))[5+kernel_size:end-4-kernel_size,kernel_size+5:end-4-kernel_size,:,:])   end end
        else
            C = zeros(size_[1], size_[2], N)
            for i in 1:N  C[:,:,i] = abs.(DSP.conv(LoG_kernel, make_padding(grey_project(I[:,:,:,i]) , kernel_size))[5+kernel_size:end-4-kernel_size,kernel_size+5:end-4-kernel_size,:])  end
        end
    elseif (ndims(I) == 5) # Dims 4 corresponds to [im_y, im_z, e, colors, N_im] 
        if per_color
            C = zeros(size_[1], size_[2], size_[3] , 3, N)
            for i in 1:N for j in 1:3 for k in size_[3] C[:,:,k,j,i] = abs.(DSP.conv(LoG_kernel, make_padding(I[:,:,k,j,i] , kernel_size))[5+kernel_size:end-4-kernel_size,kernel_size+5:end-4-kernel_size,:,:])   end end end
        else
            C = zeros(size_[1], size_[2], size_[3], N)
            for i in 1:N for k in size_[3] C[:,:,k,i] = abs.(DSP.conv(LoG_kernel, make_padding(grey_project(I[:,:,k,:,i]) , kernel_size))[5+kernel_size:end-4-kernel_size,kernel_size+5:end-4-kernel_size,:])   end end
        end
    else
        throw("Imaging Dimensions do not work in LoG")
    end

    return C
end

function contrast_std(I)
    #=
    Contrast using standard deviation for each color using
        Var(x) = E[X^2] - E[X]^2

    FIXME: Something about this is broken, if there si only item in dim 3 (multip exp) or no exposure dim the focus is different than for the multiple exposures first few (last one is equiv to 1 exp case)
    Reimplement :: if adding index to color gives nan
    =#
    mean_kernel = ones(Float32, (3,3))
    fe = 1 # Extend of kernel
    size_ = size(I)
    var = similar(I)
    if ndims(I) == 5 # If exp dimension
        N = size_[5]
        e = size_[3]
        # Compute E[X^2] - E[X]^2
        for i in N for j in e var[:,:,j,:,i] = abs.(imfilter(I[:,:,j,:,i].^2, mean_kernel)[2*fe+1:end-2*fe,2*fe+1:end-2*fe,:] - imfilter(I[:,:,j,:,i], mean_kernel)[2*fe+1:end-2*fe,2*fe+1:end-2*fe,:].^2) end end
        # Return the standard deviation (Divide by 9 because of mean kernel)
        return sqrt.((var .+ 1e-12)./ 9)
    else # If no exp dimension
        N = size_[4]
        for i in N var[:,:,:,i] = abs.(imfilter(I[:,:,:,i].^2, mean_kernel)[2*fe:end-2*fe-1,2*fe:end-2*fe-1,:] - imfilter(I[:,:,:,i], mean_kernel)[2*fe+1:end-2*fe,2*fe+1:end-2*fe,:].^2) end
        return sqrt.((var .+ 1e-12)./ 9)
    end
end


function saturation(I)
    #=
    Computed for each pixel as std between colors
    =# 
    size_ = size(I)
    N = size_[4]
    if ndims(I) == 5 # If exp dimension
        C = zeros(size_[1], size_[2],size_[3], N)
        for i in 1:N
            # Sat computed as std of color channels
            R = @view I[:,:,:,1,i]
            G = @view I[:,:,:,2,i]
            B = @view I[:,:,:,3,i]
            μ = (R+G+B)/3
            C[:,:,i] = @inbounds sqrt.(((R - μ).^2 + (G - μ).^2 + (B - μ).^2)/3)
        end
    else# If no exp dimension
        C = zeros(size_[1], size_[2], N)
        for i in 1:N
            # Sat computed as std of color channels
            R = @view I[:,:,1,i]
            G = @view I[:,:,2,i]
            B = @view I[:,:,3,i]
            μ = (R+G+B)/3
            C[:,:,i] = @inbounds sqrt.(((remove_singleton(I[:,:,1,i]) - μ).^2 + (remove_singleton(I[:,:,2,i])- μ).^2 + (remove_singleton(I[:,:,3,i]) - μ).^2)/3)
        end
    end
    return C
end

function well_exposedness(I) 
    size_ = size(I)
    N = size_[4]
    sig = 2
    if ndims(I) == 5 # If exp dimension
        C = zeros(size_[1], size_[2],size_[3], N)
        for i in 1:N
            # In matlab implemented as:
            #R = exp(-0.5*(I[:,:,1,i] .- 0.5).^2/sig.^2);
            # as this is the gaussian we implement as 
            R = @inbounds  exp.(-0.5*((I[:,:,:,1,i] .- 0.5).^2)/(sig^2));
            G = @inbounds exp.(-0.5*((I[:,:,:,2,i] .- 0.5).^2)/(sig^2));
            B = @inbounds exp.(-0.5*((I[:,:,:,3,i] .- 0.5).^2)/(sig^2));
            C[:,:,:,i] = @inbounds R.*G.*B
        end
    else
        C = zeros(size_[1], size_[2], N)
        for i in 1:N
            # In matlab implemented as:
            #R = exp(-0.5*(I[:,:,1,i] .- 0.5).^2/sig.^2);
            # as this is the gaussian we implement as 
            R = @inbounds exp.(-0.5*((I[:,:,1,i] .- 0.5).^2)/(sig^2));
            G = @inbounds exp.(-0.5*((I[:,:,2,i] .- 0.5).^2)/(sig^2));
            B = @inbounds exp.(-0.5*((I[:,:,3,i] .- 0.5).^2)/(sig^2));
            C[:,:,i] = @inbounds R.*G.*B
        end
    end
    return C
end

function Grey_lstar(I)
    #=
    Greyscale lstar
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
end

function Grey_Mean(I)
    return mean(I, dims = 3)[:,:,1]
end


#=
Nearest Neighbor Deconvolution
=#


function NearestNeighborDeconvolution(Imstack,weights, alpha=1.0, sigma_estimate=nothing)
    #=
    Applies nearest neighbor deconvolution algorithm
    alpha - strength of deblurring
    sigma_estimate - Estimated nosie
    
    Overwrites images one at a time, starting at the last image and keeping a copy of the current so that the next one can run
    =# 
    if isnothing(sigma_estimate) sigma_estimate = NoiseEstimate(Imstack[:,:,:,trunc(Int,size(Imstack)[4]/2)]) end
    fe = 4 # Filter extent
    filter = GaussianKernel(2*fe+1, sigma_estimate)
    # Select first image as tmp
    tmp_old = Imstack[:,:,:,1]
    # Invert weights matrix so that larger values get removed at less in focus locaitons
    weights = 1 .- weights
    # 4th dim contains individual img pixels
    for k in 2:size(Imstack)[4]-1
        tmp_new = @inbounds  (1+alpha)*Imstack[:,:,:,k] 
        tmp_new .-= @inbounds weights[:,:,:,k-1] .* (alpha/2) .* abs.(imfilter(tmp_old, filter, "replicate")[2*fe+1:end-2*fe,2*fe+1:end-2*fe,:]) 
        tmp_new .-= @inbounds weights[:,:,:,k+1] .* (alpha/2) .* abs.(imfilter(Imstack[:,:,:,k+1], filter, "replicate")[2*fe+1:end-2*fe,2*fe+1:end-2*fe,:])
        # In case of overflow (or under)
        tmp_new[tmp_new.>typemax(eltype(Imstack))] .= typemax(eltype(Imstack))
        tmp_new[tmp_new.<typemin(eltype(Imstack))] .= typemin(eltype(Imstack))
        # Assign current as old and computed as current and repeat
        tmp_old = Imstack[:,:,:,k]
        # Add back to image list as UInt16
        #if eltype(Imstack) == UInt16
        #    Imstack[:,:,:,k] = trunc.(eltype(Imstack), tmp_new)
        #else
        Imstack[:,:,:,k] = tmp_new
    end
    return Imstack
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






#= Bilateral Filter with contrast edge weights

Histogram Equalization -- Implemented - probably wrong
Bilater Filter extended -- paper proposed kernel Implemented, but returning to the original histogram distribution is not
                            --> Need to make sense of conovlutions between Histogram distributions

For now Failed project, if reviseted:
https://jivp-eurasipjournals.springeropen.com/articles/10.1186/s13640-015-0082-5#additional-information

Can be implemented without the paper defined kernel addition --> that actually works (I think)

usable if only  BilaterFilter(pyr_lvl,sigma_space, sigma_intensity,grey_project) is used    
            -> But computationally expensive and effects arent particularly strong

=#

function LapLevelDenoising(pyr_low, pyr_high, grey_project, lambda, H_x=nothing)
    #=
    high band image L1, and lower band image L2
    Hx --> Histogram of the original image
    grey_project --> Grey projector
    lambda --> Combination magnitude

    =# 
    p(k,κ, x, γ) = k*exp(-κ*(abs(x)^γ)) # κ∈[0.001,3] and γ∈[0.02,1.5]

    # Apply standard BLF filter to low band with spacial var 1.8 and computed noise variance
    noise_sigma = NoiseEstimate(pyr_low)
    L2_filtered = BilaterFilter(pyr_low, 1.8, noise_sigma,grey_project)
    # Make edgemap on upsampled Low band image
    odd = 2 .*size(L2_filtered) .- size(pyr_high)[1:2]
    L2_edge_map = MakeEdgeMap(upsample(L2_filtered, odd, nothing))
    # Make sigma estimate for kernel based on L1_hist
    L1_sigma = similar(L2_edge_map)
    L1_sigma[L2_edge_map .== 1] .= 2*sqrt(2)*noise_sigma
    L1_sigma[L2_edge_map .== 0] .= 4*sqrt(2)*noise_sigma
    L1_filtered = similar(pyr_high)
    for i in 1:3
        L1_filtered[:,:,i] = BilaterFilter(pyr_high[:,:,i], 1.8, noise_sigma,grey_project)
    end
    # L1 hat
    L1_adaptive_filtered = similar(L1_filtered)
    no_grey(x) = x # Dummy grey project
    for i in 1:3
        L1_adaptive_filtered[:,:,i] = BilaterFilter(pyr_high[:,:,i], 1.8, noise_sigma,no_grey,L1_sigma)
    end
    # Generate Histograms -- FIXME : I dont understand 
    # Returns histogram not equalized image
    #H_y = HistEqualization(L1_adaptive_filtered, true)
    #H_v = [p(k, κ, x, γ) for x in 1:length(H_y)]
    #H_v = H_v./sum(H_v)
    #H_yp = H_y
    #H_yp[H_y .< 0] = 0
    #H_yn = H_y
    #H_yn[H_y .> 0] = 0
    #H_r = minimum(abs(H_yp - DFT.conv(H_x, H_v))^2 + abs(H_yn + DFT.conv(H_x, H_v))^2) #+ c*R(H_r)) # I dont understand how to implement this
    # Now map the Hr histogram to the image L1_adaptive_filtered then replace the last term below
    return lambda .* L1_filtered .+ (1-lambda) .* L1_adaptive_filtered 
end


function BilaterFilter(pyr_lvl,sigma_space, sigma_intensity,grey_project,sigma_pixelbandwith=nothing)
    #=
    Bilateral Filter from : https://jivp-eurasipjournals.springeropen.com/articles/10.1186/s13640-015-0082-5#additional-information

    The standard Bilateral filter is defined as the combined domain and range filtering 
    Where domain acts as a standard average kernel, and range filter uses a similarity filter to filter out any averaging that would cause blur
    Ie at a strong transition average ie [0,0,0] [255,255,255] [255,255,255] , centered on 255 would ignore the 0 values to maintain sharpness

    This works, but the total processing time is 6 times longer than normal (did not test how long this function takes)
    =#  # FIXME: Finish the goddamn algorithm, never actually added the third component
    pyr_lvl = grey_project(pyr_lvl)
    size_ = size(pyr_lvl)
    kernel_size = trunc(Int, 2*sigma_space+1)
    half_kernel_size = trunc(Int, kernel_size / 2)
    result = zeros(size(pyr_lvl))
    W = 0
    pyr_padded = make_padding(pyr_lvl, half_kernel_size+1)
    if !isnothing(sigma_pixelbandwith)
        # Do histogram equalization and padding
        histeq = HistEqualization(grey_project(pyr_lvl))
        histeq = make_padding(pyr_lvl, half_kernel_size)
    end
    # Make the distance kernel
    distance_kernel = [exp(-(x^2+y^2)/2*(sigma_space^2)) for x in -half_kernel_size:half_kernel_size+1, y in -half_kernel_size:half_kernel_size+1]
    #iterate the kernel
    for i in 1:size_[1]
        for j in 1:size_[2]
            subsection = pyr_padded[i:i+kernel_size, j:j+kernel_size]
            similarity_kernel = Gaussian((subsection .- pyr_padded[i,j,:]).^2, sigma_intensity)
            # Paper kernel - do hist equalization then apply same as similarity
            if !isnothing(sigma_pixelbandwith)
                subhist = histeq[i:i+kernel_size, j:j+kernel_size]
                edge_kernel = Gaussian((subhist .- histeq[i,j,:]).^2, sigma_pixelbandwith[i, j])
                bilateral_kernel = similarity_kernel .* distance_kernel .* edge_kernel
            else
                bilateral_kernel = similarity_kernel .* distance_kernel
            end
            norm = sum(bilateral_kernel)
            # Write to result
            result[i, j] = sum(bilateral_kernel .* subsection) / norm
        end
    end
    return result
end


function HistEqualization(I,ret_hist=false) # This is wrong
    #=
    Performs histogram Equalization for Bilateral Filter
        I -> Single image to be made into histogram
    =#
    # Check if data preprocessed float or uint --> will almost always be flaot 0-1
    orig_type = eltype(I)
    if eltype(I) ∉ (UInt8, UInt16)
        max_ = maximum(I) # FIXME: Check what type is actually being passed
        min_ = minimum(I)
        I = trunc.(UInt16,((I .- min_) ./ (max_-min_)) .* (2^16-1))
    elseif eltype(I) ∈ (UInt8)
        I = trunc.(UInt16, I./(2^8-1).* (2^16-1))
    end
    bincount = countmap(I) # Returns dictionary
    bincount = [get(bincount, i, 0) for i in 0:(2^16-1)] # Index list sorted by px val
    bincount_norm = bincount/sum(bincount)
    # Early Return
    if ret_hist return bincount_norm end
    chistogram_array = cumsum(bincount_norm)
    # Lookup table:
    transform_map = trunc.(UInt16, chistogram_array .* (2^16-1))
    # Replace px with lookup table
    replace_px(x) = transform_map[x+1]
    I = replace_px.(I)
    # Now convert back fix scale and return
    if orig_type ∈ (UInt8, UInt16)
        return trunc.(orig_type, I ./ (2^16-1) .* (max_-min_) .+ min_)
    else
        return I ./ (2^16-1) .* (max_-min_) .+ min_
    end
end