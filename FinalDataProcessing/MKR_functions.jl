# All functions regarding the MKR algorithm will be stored here
module MKR_functions

using DSP
###                                 Pyramid Functions

function Pyramid_Filter()
    return [.0625, .25, .375, .25, .0625]
end

function Gaussian_Pyramid(I,nlev=nothing; filter = Pyramid_Filter())
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
    # Make copy, assigning now messes up some type stuff, really weird error
    J = I
    for l in 2.0:nlev
        I=downsample(I,filter)
        pyr[l] = I
    end
    pyr[1.0] = J
    return pyr
end

function Laplacian_Pyramid(I,nlev=nothing;filter = Pyramid_Filter())
    size_ = size(I)
    r = size_[1]
    c = size_[2]

    if (isnothing(nlev))
        nlev = floor(log(min(r,c)) / log(2))
    end
    J = convert(Array{Float32}, I)
    # Recursively build pyramid
    pyr = Dict()
    for l in (1:nlev-1)
        I = downsample(J, filter)
        odd = 2 .*size(I) .- size(J)
        # Store difference between image and upsampled low pass version
        pyr[l] = J-upsample(I, odd, filter)
        J = I
    end
    pyr[nlev] = J
    return pyr
end
using Images
function Reconstruct_Laplacian_Pyramid(pyr; filter = Pyramid_Filter())
    #=
    Takes pyramid and makes image
    =#
    nlev = length(pyr)
    # Start with low pass residual
    #printstyled("Making Pyramid Video! In MKR_functions.Reconstruct_Laplacian_Pyramid\n", color=:red)
    #if !isdir("/SaveSpot/FakeBee2/0_35828_32000/vid")
    #    mkdir("/SaveSpot/FakeBee2/0_35828_32000/vid")
    #end
    R = pyr[nlev]
    for l in ((nlev-1):-1:1)
        # Upsample, add to current level
        odd = 2 .*size(R) .- size(pyr[l])
        R = pyr[l] + upsample(R,odd,filter)
        #Images.save("/SaveSpot/FakeBee2/0_35828_32000/vid/$l.png", R./maximum(R))
        if any(isnan.(R)) println("Laplacian reconst at level $l has $(sum(isnan.(R))) Nan values") end   
    end
    return R
end

function GenerateEmptyPyramids(w_n, h_n, nlev, N)
    pyr = Dict()
    pyr_Weight = Dict()
    Weight_mat = Float32.(zeros(w_n,h_n, 3, N))
    for l in (1:nlev)
        pyr[l] = Float32.(zeros(w_n, h_n, 3, N))
        pyr_Weight[l] = Array{Float32}(undef, (w_n, h_n, 3, N))
        w_n2, h_n2 = trunc(Int, w_n/2), trunc(Int, h_n/2)
        if (w_n2 *2 - w_n != 0)  w_n = w_n2+1 
        else w_n = w_n2 end
        if (h_n2 *2 - h_n != 0)  h_n = h_n2+1 
        else h_n = h_n2 end
    end
    return pyr, pyr_Weight, Weight_mat
end


###                             Resampling functions

function downsample(I, filter,subsample=2)
    """Original
    removed as I may have implemented filtering only along rows, doing now what I know works """
    ## Apply paded filter
    #if ndims(filter) == 1  filter=filter*filter' end
    #extended = imfilter(I,filter)
    ## remove padding 
    #fe = trunc(Int,size(filter)[1]/2)
    #extended=extended[1+2*fe:end-2*fe, 1+2*fe:end-2*fe,:]
    ## return every other element
    #return extended[1:2:end,1:2:end,:]
    #printstyled("Downample\n", color=:red)
    fe = trunc(Int,size(filter)[1]/2)
    # Do convolution (to avoid aliasing) and subsample
    R = DSP.conv(filter*filter', I)[fe+1:subsample:end-fe, fe+1:subsample:end-fe,:]
    return R
end

function upsample(I,odd, filter)
    # Make zero array
    r,c = size(I)[1:2].*2
    R =zeros(r,c,3)
    fe = trunc(Int,size(filter)[1]/2)
    R[1:2:r, 1:2:c, :] = I 
    R[2:2:r, 1:2:c, :] = I 
    R[1:2:r, 2:2:c, :] = I 
    R[2:2:r, 2:2:c, :] = I 
    ## Fix borders in case we need odd sized array
    if odd[1] == 1 R = R[1+odd[1]:end,1:end,:] end
    if odd[2] == 1 R = R[1:end,1+odd[2]:end,:] end
    # Convolution to smooth the image to avoid any blocky features
    R = DSP.conv(filter*filter', R)

    R = R[fe+1:end-fe, fe+1:end-fe,:]

    """
    Old Mistakes:
    I never assigned the output of DSP to R, so the filter was never run on upsample
    Make padding increased dim by (2,2,0)
    DSP increases size by (2,0,0)
    So I assume DSP only acts on the first axes and thats why I have it twice, once for row once for column
    Upsample filter is 1D
    Original implementation (for future reference, everything after odd):
    R = make_padding(R, fe) 
    DSP.conv(R, filter)
    R = make_padding(R, fe) 
    DSP.conv(R, filter)
    multip =2 
    R = R[multip*fe+1:end-multip*fe, multip*fe+1:end-multip*fe,:]

    """

    return R
end

function imfilter(mono, h)
    #=
                No longer used

    Naming and args from matlab only mono and h used
    mono -- greyscale image
    h -- kernel
    
    Implemented using DSP convolution, since we want replicate we make a new array with extra columns and rows depending on filter size
    to achieve the replicate feature, the central portion of the array is a view of the original to save space
    =#
    # Make larger array 
    kernel_extend = trunc(Int,size(h)[1] / 2)
    extended = make_padding(mono, kernel_extend) 

    return DSP.conv(extended, h)
end


function make_padding(mono, kernel_extend)
    #=
    No longer used (except for imfilter, which is no longer used)

    Adds padding on each side of size kernel_extend
    Defined for 3D arrays (im_dim_1, im_dim_2, [colors]) 
    =#
    size_ = size(mono)
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
    return extended
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