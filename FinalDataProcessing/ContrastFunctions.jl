# Functions to compute image contrast
module ContrastFunctions

include("Datastructures.jl")
import .Datastructures
import DSP

# Local Standard deviation of color channels
function color_STD(dat, pp::Main.Datastructures.ProcessingParameters)
    mean_kernel = ones(Float32, (3,3))
    fe = 1 # Extend of kernel
    cont = pp.precision.(similar(dat))
    for c in (1:3)
        cont[:,:,c] = abs.(imfilter(dat[:,:,c].^2, mean_kernel)[2*fe:end-2*fe-1,2*fe:end-2*fe-1] - imfilter(dat[:,:,c], mean_kernel)[2*fe+1:end-2*fe,2*fe+1:end-2*fe].^2)
    end
    return cont
end

# Lagrangian of Gaussian Greyscaler
function LoG(I, pp::Main.Datastructures.ProcessingParameters)
    #=
    Contrast using Lagrangian of Gaussian
    I -> Image
    grey -> greyscale projector (if requiredd)
    =#
    # Lagrangian of Gaussian 1.4σ
    LoG_kernel = [[0,1,1,2,2,2,1,1,0] [1,2,4,5,5,5,4,2,1] [1,4,5,3,0,3,5,4,1] [2,5,3,-12,-24,-12,3,5,2] [2,5,0,-24,-40,-24,0,5,2] [2,5,3,-12,-24,-12,3,5,2] [1,4,5,3,0,3,5,4,1] [1,2,4,5,5,5,4,2,1] [0,1,1,2,2,2,1,1,0] ]
    kernel_size = trunc(Int, size(LoG_kernel)[1]/2)
    return repeat(abs.(DSP.conv(LoG_kernel, pp.Greyscaler(pp.precision.(I)))[kernel_size+1:end-kernel_size,kernel_size+1:end-kernel_size,:]),outer=[1,1,3])
end

end #module