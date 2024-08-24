# All functions to scale image data will be here
module ScalingFunctions

function exp_scaling_contrast(x)
    #=
    Function to scale contrast weights,
    LaTeX string: x^{4}\left(e^{x\ln\left(2\right)}-1\right)
    =#
    return x^4*exp(x*log(2))-1
end


function removeBlackpoint(img, blackpoint)
    # Remove blackpoint
    for i in 1:3
        bool_array = img[:,:,i] .<= blackpoint[i]
        img[:,:,i][bool_array] .= 0
        img[:,:,i][.!bool_array] .-= blackpoint[i]
    end
    return img
end


function ScaleWeightMatrix(W_mat,epsilon=1e-10)
    # Set zero point plus epsilon
    for i in 1:3
        W_mat[:,:,i,:] .-= minimum(W_mat[:,:,i,:])
        W_mat[:,:,i,:] .+= epsilon
    end
    # And scale 0 to 1 such that each pixels color sums to 1
    N = size(W_mat,4)
    W_mat ./= repeat(sum(W_mat,dims=4), outer=[1,1,1,N])
    
    return W_mat
end

end # module