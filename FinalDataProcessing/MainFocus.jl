# The code to fuse images will be located here
include("Datastructures.jl")
using .Datastructures
include("./IO_dp.jl")
import .IO_dp

include("./ContrastFunctions.jl")
import .ContrastFunctions
include("./Grey_Projectors.jl")
import .GreyProjectors
include("./ImageFusion.jl")
import .ImageFusion

using Base.Threads
using Images
using IterTools

println("Loaded")

path = "/Images/img_2/"
save_path = "/SaveSpot/FakeBee/"
blackpoint = [0,0,0]
contrast_precision = Float32 
width = 3040
height = 4056
debug = false
#TODO: When loading the intermediaries the dimensions of the loaded image are inverted -> Why?
pp = Datastructures.ProcessingParameters(contrast_precision, ContrastFunctions.LoG, GreyProjectors.lstar, blackpoint, path, save_path,width, height, debug)

# 10 images works great, 30 min per set for 4k, I just noticed its using so much because im running this in 20 threads
max_images = 5

# TODO: Implement GenerateImageIgnoreList from IO_dp.jl once implmeneted in microscope --> Test methods
#           Only generates indexing array for ignored vals
#           Filter out fnames should be easiest

function FocusFusion(parameters::Datastructures.ProcessingParameters,max_images::Int,tst::Bool=false)
    # Print process ID for debugging purposes
    println("PID: $(getpid())")
    if tst # When profiling need to run once to initialize
        return
    end
    # Fix paths
    parameters.path         = IO_dp.FixPath(parameters.path)
    parameters.save_path    = IO_dp.FixPath(parameters.save_path)

    # Epsilons in case of Nans (assumed) resulting from the weight matrix scaling
    epsilons = [1e-12,1e-10, 1e-8, 1e-6]
    printstyled("Running with $(nthreads()) threads\n", color=:blue)
    # Number of times fusion may fail (needs the same number of epsilons)
    allowed_fail= 4 
    # Grab image identifiers
    ImagingGrid = IO_dp.GrabIdentifiers(parameters.path)
    #Temp override
    #ImagingGrid.exp = ["NoIR"]
    #ImagingGrid.y =[47510]
    #ImagingGrid.z = [12678]
    failed_im = []
    # Iterate the imaging grid
    total = length(ImagingGrid.y)*length(ImagingGrid.z)*length(ImagingGrid.exp)
    counter = 0
    
    ignore = [IO_dp.GenerateFinalFileName(0,89571,48000), IO_dp.GenerateFinalFileName(24785,89571,48000), IO_dp.GenerateFinalFileName(49571,89571,48000)]
    
    # Parse meta file for contrast values if available and create filtering array
    contrast_max, contast_min, contrast_mean,indexing_array = nothing,nothing,nothing,nothing
    try
        if isfile(joinpath(parameters.path, "meta.txt"))
            println("Loading meta.txt to filter images by contrast")
            contrast_max, contast_min, contrast_mean = IO_dp.ParseMetaFile(joinpath(parameters.path, "meta.txt"))
            indexing_array = IO_dp.GenerateImageIgnoreListContrast(contrast_max, contast_min, contrast_mean, cont_method=1)
        end
    catch e 
        printstyled("Failed to load meta file with error", color =:red)
        println(e)
    end

    for ei in ImagingGrid.exp
    for yi in ImagingGrid.y
    for zi in ImagingGrid.z
        counter += 1
        final_name = IO_dp.GenerateFinalFileName(yi,zi,ei)
        success = false
        # Check file doesnt exist, we arent in debug, and check if the file is to be ignored (missing data)
        if ((!isfile("$(save_path)$(final_name)") || parameters.debug) && !(final_name in ignore))
            start = time()
            println("Processing image $(counter) out of $(total)")
            println("Current Focus Name: $(final_name)")
            # Generate file names
            if isnothing(indexing_array)
                fnames = [IO_dp.GenerateFileName(xi,yi,zi,ei) for xi in ImagingGrid.x]
            else # Filter for contrast values
                fnames = [IO_dp.GenerateFileName(xi,yi,zi,ei) for xi in ImagingGrid.x if indexing_array[xi,yi,zi]]
            end
            # Filter based on files available
            fnames =  [i for i in fnames if isfile(joinpath(parameters.path, i)) ]
            # Run stacking for yze postion and split them as too much ram is used
            println("Files remaining after filter $(length(fnames))")
            println("Starting MKR fusion")
            counter_ = 0
            #Temp directory as it uses too much ram and on harddrive since MKR by default sources from the harddrive path
            if !isdir(joinpath(parameters.path, "$(yi)_$(zi)_$(ei)")) 
                mkdir(joinpath(parameters.path, "$(yi)_$(zi)_$(ei)"))
            end
            for batch in IterTools.partition(fnames, max_images)
                fail_count = 1
                while fail_count <= allowed_fail 
                    if !isfile(joinpath(parameters.path, "$(yi)_$(zi)_$(ei)", "im_$(counter_).png")) || parameters.debug
                        try # In case of some grid element not being imaged
                            image = ImageFusion.MKR(batch, parameters, epsilons[fail_count])
                            # Save Image check for nans
                            if any(isnan.(image))
                                printstyled("Warning: Image $(joinpath(parameters.path, "$(yi)_$(zi)_$(ei)", "im_$(counter_).png")) contains NaNs. Retrying...\n", color=:red)
                                fail_count += 1
                            else
                                Images.save(joinpath(parameters.path, "$(yi)_$(zi)_$(ei)", "im_$(counter_).png"), image)
                                counter_ += 1
                                success = true
                                break
                            end
                        catch e
                            if isa(e, CompositeException)
                                println("File didnt exist, stack added to failed_im tracker")
                                break # Assume its a arguemnt error to load file
                            else
                                rethrow() # something else
                            end
                        end
                        
                    else # Terminate if already exists
                        printstyled("$(joinpath(parameters.path, "$(yi)_$(zi)_$(ei)", "im_$(counter_).png")) already exists, skipping\n", color=:yellow)
                        counter_ += 1
                        success = true
                        break
                    end
                end
                if success
                    printstyled("Completed $(joinpath(parameters.path, "$(yi)_$(zi)_$(ei)", "im_$(counter_).png"))\n", color=:green)
                    printstyled("Time elapsed: $(time() - start)\n", color=:blue)
                else
                    printstyled("Failed $(joinpath(parameters.path, "$(yi)_$(zi)_$(ei)", "im_$(counter_).png"))\n", color=:red)
                    printstyled("Time elapsed: $(time() - start)\n", color=:red)
                    failed_im = push!(failed_im, final_name)
                end
            end
            fail_count  = 1
            while fail_count <= allowed_fail
                #TODO: The width and height are the other way around here for some reason when loading the images
                parameters_tmp = deepcopy(parameters)
                _width = parameters.width
                _height = parameters.height
                parameters_tmp.width = _height
                parameters_tmp.height = _width
                success = false
                # And now fuse the remaining images (TODO this will at some point be more than N images, make this dynamically work)
                image = ImageFusion.MKR([joinpath("$(yi)_$(zi)_$(ei)", "im_$(i).png") for i in 0:counter_-1] , parameters_tmp, epsilons[fail_count])
                if any(isnan.(image))
                    printstyled("Warning: Image $(final_name) contains NaNs. Retrying...\n", color=:red)
                    fail_count += 1
                else
                    # Save the image and mark as successful if no NaNs found
                    savepath = joinpath(save_path, final_name)
                    Images.save(savepath, image)
                    println("Image $(final_name) saved successfully.")
                    success = true
                    break
                end
            end
            if success
                printstyled("Completed $(final_name)\n", color=:green)
                printstyled("Time elapsed: $(time() - start)\n", color=:blue)
            else
                printstyled("Failed $(final_name)\n", color=:red)
                printstyled("Time elapsed: $(time() - start)\n", color=:red)
                failed_im = push!(failed_im, final_name)
            end

        else
            if final_name in ignore
                printstyled("$(final_name) to ignore as specified\n", color=:yellow)
            else
                printstyled("$(final_name) already exists, skipping\n", color=:yellow)
            end
        end
    end
    end
    end
    println("Failed images: $(failed_im)")
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
    FocusFusion(pp, max_images)
end