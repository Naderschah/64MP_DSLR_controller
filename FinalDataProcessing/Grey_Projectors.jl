# Holds all implemented grey projectors
module GreyProjectors

# For very red heavy images
function Mean_No_red(I)
    return (I[:,:,2] + I[:,:,3])./2
end

# l-star Greyscaler
function lstar(I)
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

# Mean Grey Scaler
function Mean(I)
    return mean(I, dims = 3)[:,:,1]
end

end #module