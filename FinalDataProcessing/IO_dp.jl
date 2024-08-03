# Functions regarding image loading, file name parsing, data saving will be located here


module IO_dp

include("./Datastructures.jl")
import .Datastructures

using StatsBase

include("./ImagePlacement.jl")

export GenerateFileName, GenerateFinalFileName, FixPath, GrabIdentifiers, GetFocusedIdentifiers, GetFocusedImageGrid, LoadDNGLibRaw, RowColumnIndexNaming

# Generate pre focus file naming patterns
function GenerateFileName(x,y,z,exp)
    return "$(x)_$(y)_$(z)_exp$(exp).png"
end

function GenerateFinalFileName(x,y,e)
    return "Focused_y=$(x)_z=$(y)_e=$(e).png"
end

function FixPath(path)
    if endswith(path, "/")
        return path
    else
        return path*"/"
    end
end


# Get image identifiers from directory
function GrabIdentifiers(image_directory)
    println(image_directory)
    files = [f for f in readdir(image_directory) if (!isdir(joinpath(image_directory,f)) && splitext(basename(f))[end] != ".txt")]# readdir(image_directory)
    x_y_z_exp = [split(i, "_") for i in files]
    x_y_z = [[parse(Int, String(i[1])),parse(Int, String(i[2])), parse(Int, String(i[3]))] for i in x_y_z_exp]
    # In case loading files fucks up again
    #problematic_entries = []
    #for (index, item) in enumerate(x_y_z_exp)
    #    try
    #        # Attempt to access and process the fourth index as in your original expression
    #        parsed_value = parse(Int, String(split(item[4], ".")[1])[4:end])
    #    catch e
    #        # If an error occurs, add the index and item to the problematic list
    #        push!(problematic_entries, (index, item))
    #    end
    #end
    #println(problematic_entries)
    #println(isdir(files[problematic_entries[1][1]]))

    exp = []
    exp = unique([parse(Int, String(split(i[4], ".")[1])[4:end]) for i in x_y_z_exp])

    x = unique([i[1] for i in x_y_z])
    y = unique([i[2] for i in x_y_z])
    z = unique([i[3] for i in x_y_z])
    # We will only do one yz pos for starters
    x = sort(x)
    y = sort(y)
    z = sort(z)

    return Datastructures.ImagingGrid(x,y,z,exp)
end

function GetFocusedIdentifiers(ImagingParams::Datastructures.ProcessingParameters)
    files = readdir(ImagingParams.save_path)
    f_y_z_exp = [split(i, "_") for i in files]
    f_y_z = [ endswith(i[end], ".png") ? [i[1],parse(Int, String(split(i[2],"=")[2])), parse(Int, String(split(i[3], "=")[2]))] : ["F", -1, -1] for i in f_y_z_exp]
    y = unique([i[2] for i in f_y_z])
    z = unique([i[3] for i in f_y_z])
    y = sort(y)
    if -1 in y
        deleteat!(y, findall(x->x==-1, y))
    end
    z = sort(z)
    if -1 in z
        deleteat!(z, findall(x->x==-1, z))
    end
    return y, z
end

function GetFocusedImageGrid(ImagingParams::Datastructures.ProcessingParameters,fnamefunc=GenerateFinalFileName, exp="NoIR")
    #=
    Here we jsut make an array of the image names to be loaded
    =#
    x, y = GetFocusedIdentifiers(ImagingParams)
    x = reverse(x)
    y = reverse(y)
    img_name_grid = Array{String}(undef,length(y), length(x))
    img_pos_grid = Array{Float64}(undef,length(y), length(x), 2)
    for i in eachindex(x)
        for j in eachindex(y)
            img_name_grid[j,i]  = fnamefunc(x[i],y[j],exp)
            # Below doesnt work ause needs imagingparams not processing
            #img_pos_grid[j,i,:] = GeneratePxCoordinates([x[i],y[j]], ImagingParams)
        end
    end
    return img_name_grid[end:-1:1,end:-1:1]#, img_pos_grid[end:-1:1,end:-1:1,:]
end



# Load DNG images
function LoadDNGLibRaw(path, size=(3,3040,4056))
    # Uses C API LibRaw to load image char array
    # Size should be Color, height, width
    # img is to be populated by the ccall
    img = Array{UInt16}(undef, size[1]*size[2]*size[3])
    success = Ref{Cint}(0)
    @ccall "./C_stuff/LibRaw_LoadDNG.so".LoadDNG(img::Ref{Cushort},path::Ptr{UInt8},success::Ref{Cint})::Cvoid
    # Grab the reference from mem
    #print("success : $(success[])")
    if success[] == 0
        # TODO: Find a way to actually change the value of success in the C-code for some reason i cant figuer it out
        println("\033[93m   Error loading image : $(path) \033[0m") 
        println(success)
        img .= 0
    end
    # The array is linear so we need to reshape it to get the correct data
    order = [1,3,2]
    size = size[order]
    img = permutedims(reshape(img, size), (3,2,1))
    return img
end


function RowColumnIndexNaming(ImagingParams::Datastructures.ProcessingParameters, img_name_grid=nothing)
    #=
    Function for Fiji MIST, renames from  y z coordinates to row column

    We use save path as this runs after focus stacking
    =#
    # We get the formated frid for names
    if isnothing(img_name_grid)
        img_name_grid = GetFocusedImageGrid(ImagingParams)
    end
    # And generate the names for the files
    new_names = Array{String}(undef, size(img_name_grid))
    for i in 1:size(img_name_grid,1)
        for j in 1:size(img_name_grid,2)
            #Move file to new name
            new_name = "r$(i)_c$(j).png"
            if isfile(ImagingParams.save_path*img_name_grid[i,j])
                mv(ImagingParams.save_path*img_name_grid[i,j], ImagingParams.save_path*new_name)
                println("Moving $(ImagingParams.save_path*img_name_grid[i,j]) to $(ImagingParams.save_path*new_name)")
            end
        end
    end
    return img_name_grid
end


function ParseMetaFile(path)
    f = open(path, "r")
    cont = read(f, String)
    close(f)

    cont = split(cont, "\n")
    # 1 is header
    cont = [split(cont[i], ",") for i in 2:length(cont)]
    # Change dtypes
    cont[:,1:3]  = Int.(cont[:,1:3])
    cont[:,4:end] = Float.(cont[:,4:end])
    x_val = length(unique(cont[:,1]))
    y_val = length(unique(cont[:,2]))
    z_val = length(unique(cont[:,3]))

    accel = vec(cont[:,4])
    # Just some info
    printstyled("Acceleration statistics: mean=$(mean(accel)) std=$(std(accel)) max=$(maximum(accel)) min=$(minimum(accel))", color=:yellow)

    contrast = zeros(Float, (3, x_val, y_val, z_val))
    # Populate structured array
    for i in eachindex(cont)
        contrast[:,cont[i,1],cont[i,2],cont[i,3]] = cont[i,6:8]
    end
    # And unpack
    contrast_max, contast_min, contrast_mean = Tuple(contrast[i,:,:,:] for i in 1:3)
    return contrast_max, contast_min, contrast_mean
end# TODO This doesnt work with multiple exposure, same with below

function GenerateImageIgnoreListContrast(contrast_max, contast_min, contrast_mean, cont_method=1)
    #TODO Watch out contrast vals computes mean max and min
    # Could be useful to compute stats based on mean, and then also take into account max for selection 

    # We will first generate some statistics from mean
    flattend_contrast = vec(contrast_mean)
    _mean,_median, _std, _min, _max = mean(flattend_contrast), median(flattend_contrast),std(flattend_contrast), minimum(flattend_contrast), maximum(flattend_contrast)

    prinln("_mean")
    prinln(_mean)
    prinln("_median")
    prinln(_median)
    prinln("_std")
    prinln(_std )
    prinln("_min")
    prinln(_min )
    prinln("_max")
    prinln(_max)

    # TODO: Test the filtering technique
    if cont_method == 1
        threshhold = _mean - _std
    elseif cont_method == 2
        threshhold = _median - _std
    elseif cont_method == 3
        threshhold = _max - 2*_std
    elseif cont_method == 4
        threshhold = (_min+_max)/2 - _std 
    elseif cont_method == 5
        threshhold = (_mean+_median)/2 -_std
    else
        println("Please specify a method")
    end

    printstyled("Rejecting $(round(100*(flattend_contrast .< threshhold)/length(flattend_contrast)))", color=:yellow)
    println("Threshhold $(threshhold)")

    # Return indexing array
    return contrast_mean .> threshhold
end



end # module


