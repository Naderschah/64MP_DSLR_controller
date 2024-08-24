module Kernels



function LoGKernel(n, sigma=nothing)
    if sigma == nothing sigma = n/6 end
    if !isodd(n) throw(Error("Kernel dim must be odd")) end

    kernel = zeros((n,n))
    shift = n÷2 + 1
    coeff = - 1/ (pi*sigma^4)
    for i in -n÷2:n÷2
        for j in -n÷2:n÷2
            factor = -(i^2+j^2)/(2*sigma^2)
            kernel[i+shift,j+shift] = coeff * (1 + factor)exp(factor)
        end
    end
    return kernel / sum(kernel)
end


function GaussianKernel(n, sigma=nothing)
    if sigma == nothing sigma = n/6 end
    if !isodd(n) throw(Error("Kernel dim must be odd")) end
    kernel = zeros((n))
    shift = n÷2 + 1
    coeff = - 1/ (2*pi*sigma^2)
    for i in -n÷2:n÷2
        kernel[i+shift] = coeff * exp(-(i^2)/(2*sigma^2))
    end
    return kernel / sum(kernel)
end




end # module