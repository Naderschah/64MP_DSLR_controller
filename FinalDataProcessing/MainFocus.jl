# The code to fuse images will be located here
include("./IO_dp.jl")
import .IO_dp
include("Datastructures.jl")
import .Datastructures
include("./ContrastFunctions.jl")
import .ContrastFunctions
include("./Grey_Projectors.jl")
import .GreyProjectors
include("./ImageFusion.jl")
import .ImageFusion

using Base.Threads
using Images



path = "/media/felix/3677421f-5daf-4ea7-ba33-31b09d11edcf/Images/SmallWasp/"
save_path = "/mnt/rapid_storage_2/SmallWasp/"
blackpoint = [0,0,0]
contrast_precision = Float32 
width = 4056
height = 3040
debug = false
pp = Datastructures.ProcessingParameters(contrast_precision, ContrastFunctions.LoG, GreyProjectors.lstar, blackpoint, path, save_path,width, height, debug)



function FocusFusion(parameters::Datastructures.ProcessingParameters,tst::Bool=false)
    if tst # When profiling need to run once to initialize
        return
    end
    # Fix paths
    parameters.path         = IO_dp.FixPath(parameters.path)
    parameters.save_path    = IO_dp.FixPath(parameters.save_path)

    printstyled("Running with $(nthreads()) threads\n", color=:blue)
    # Grab image identifiers
    ImagingGrid = IO_dp.GrabIdentifiers(parameters.path)
    # Iterate the imaging grid

    for ei in ImagingGrid.exp
    for yi in ImagingGrid.y
    for zi in ImagingGrid.z
        final_name = IO_dp.GenerateFinalFileName(yi,zi,ei)
        # Check file doesnt exist
        if !isfile("$(save_path)$(final_name)") || parameters.debug
            start = time()
            # Generate file names
            fnames = [IO_dp.GenerateFileName(xi,yi,zi,ei) for xi in ImagingGrid.x]
            # Run stacking for yze postion
            image = ImageFusion.MKR(fnames, parameters)
            # Save Image
            Images.save("$(save_path)$(final_name)", image)
            
            printstyled("Completed $(final_name)\n", color=:green)
            printstyled("Time elapsed: $(time() - start)\n", color=:blue)
        else

            printstyled("$(final_name) already exists, skipping\n", color=:yellow)

        end
    end
    end
    end
end

# Profiling
using Profile
#using ProfileView # using porfile view
#ProfileView.set_theme!(:dark)
#ProfileView.@profview FocusFusion(pp)
using PProf # Using PProf

if isinteractive()
    # Profilling - Runs only in repl
    
    Profile.clear()
    @profile FocusFusion(pp)
    pprof()
else
    FocusFusion(pp)
end