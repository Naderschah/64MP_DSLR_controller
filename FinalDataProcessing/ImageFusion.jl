module ImageFusion

include("Datastructures.jl")
import .Datastructures
include("MKR_functions.jl")
import .MKR_functions
include("./IO_dp.jl")
import .IO_dp
include("ScalingFunctions.jl")
import .ScalingFunctions

# Merten Kautz van Reeth image fusion
function MKR(fnames, pp::Main.Datastructures.ProcessingParameters)

    # N Images
    N = size(fnames,1)
    # Number of pyramid levels
    nlev = floor(log(min(pp.height ,pp.width)) / log(2))

    # Prealocate data structures
    pyr, pyr_Weight, Weight_mat = MKR_functions.GenerateEmptyPyramids(pp.width, pp.height, nlev, N)

    Threads.@threads for x in eachindex(fnames)

        # Load image
        img = IO_dp.LoadDNGLibRaw(pp.path*fnames[x], (3,pp.height,pp.width))
        # Remove blackpoint
        img = ScalingFunctions.removeBlackpoint(img, pp.blackpoint)
        # Scale 0 to 1
        img = img ./ typemax(eltype(img))

        # Compute Contrast
        Weight_mat[:,:,:,x] = pp.ContrastFunction(img, pp)
        # Generate image Pyramid
        img_pyr = MKR_functions.Laplacian_Pyramid(img, nlev)
        # Assign to final pyramid
        for l in (1:nlev) @inbounds pyr[l][:,:,:,x] = img_pyr[l]  end

        println("Added image $(fnames[x])")

    end
    # Scale weight matrix such that it is in the range 0 to 1 and each pixel adds up to 1
    Weight_mat = ScalingFunctions.ScaleWeightMatrix(Weight_mat)

    # Create pyramid weight matrix
    Threads.@threads for i = 1:N
        tmp = MKR_functions.Gaussian_Pyramid(Weight_mat[:,:,:,i])
        for l in (1:nlev)
            pyr_Weight[l][:,:,:,i] = tmp[l]
        end
    end
    # Free weight matrix memory
    Weight_mat = nothing

    # Create final pyramid
    fin_pyr = Dict()
    Threads.@threads for l in (1:nlev)
        @inbounds fin_pyr[l] = sum(pyr_Weight[l][:,:,:,:] .* pyr[l][:,:,:,:], dims=4)[:,:,:,1]  
    end
    
    # Reconstruct image and return
    return clamp.(MKR_functions.Reconstruct_Laplacian_Pyramid(fin_pyr), 0, 1)
end

end # module