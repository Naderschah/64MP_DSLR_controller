# Functions regarding image loading, file name parsing, data saving will be located here


module IO_dp

include("./Datastructures.jl")
import .Datastructures

using StatsBase
using DelimitedFiles
using PrettyTables

include("./ImagePlacement.jl")

export GenerateFileName, GenerateFinalFileName, FixPath, GrabIdentifiers, GetFocusedIdentifiers, GetFocusedImageGrid, LoadDNGLibRaw, RowColumnIndexNaming

# Generate pre focus file naming patterns
function GenerateFileName(x,y,z,exp)
    return "$(x)_$(y)_$(z)_exp$(exp).hdf5"
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
    cont = [split(cont[i], ",") for i in 2:length(cont) if cont[i] != ""]
    try
        cont = stack(cont, dims=1)
    catch
        cont = stack(cont[1:end-1],dims=1)
    end
    # Turn into array
    xyz  = map(x->parse(Int64,x),cont[:,1:3])
    cont = map(x->parse(Float64,x),cont) 
    # We now need to make an indexing table of the xyz list
    x_vals = sort(unique(xyz[:,1]))
    y_vals = sort(unique(xyz[:,2]))
    z_vals = sort(unique(xyz[:,3]))
    # Now we make an indexing dict
    x_indexing = Dict(x_vals[i] => i for i in eachindex(x_vals))
    y_indexing = Dict(y_vals[i] => i for i in eachindex(y_vals))
    z_indexing = Dict(z_vals[i] => i for i in eachindex(z_vals))

    contrast = zeros(Float64, (3, length(x_vals), length(y_vals), length(z_vals)))
    # Populate structured array
    for i in axes(cont, 1) 
        contrast[:,x_indexing[xyz[i,1]],y_indexing[xyz[i,2]],z_indexing[xyz[i,3]]] = cont[i,5:7]
    end
    # And unpack
    contrast_max = contrast[1,:,:,:]
    contast_min = contrast[2,:,:,:]
    contrast_mean = contrast[3,:,:,:]
    return contrast_max, contast_min, contrast_mean, (x_indexing,y_indexing,z_indexing)
end# TODO This doesnt work with multiple exposure, same with below

function GenerateImageIgnoreListContrast(contrast_max, contast_min, contrast_mean, cont_method=1)
    # We will first generate some statistics from mean
    flattend_contrast = vec(contrast_mean)
    _mean,_median, _std, _min, _max = mean(flattend_contrast), median(flattend_contrast),std(flattend_contrast), minimum(flattend_contrast), maximum(flattend_contrast)
    
    # Generate pretty print table for easier adjustment of method during processing
    threshhold = ["Threshhold",_mean -0.5*_std, _median - _std, _max - 2*_std, (_min+_max)/2 - _std, (_mean+_median)/2 -_std]
    percrejected = Any["Rejected (%)"]
    percrejected2 = [100*sum(flattend_contrast .< i)/length(flattend_contrast) for i in threshhold[2:end]]
    append!(percrejected, percrejected2)

    header = ["","Mean-0.5STD", "Median-STD", "Max-2STD", "(Min+Max)/2 - STD", "(Mean+Median)/2-STD"]
    hl_current = PrettyTables.Highlighter(
           (data, i, j) -> (j == cont_method+1) && (i != 1),
           PrettyTables.crayon"blue bold"
       );
    hl_column = PrettyTables.Highlighter(
        (data, i, j) -> (j == 1),
        PrettyTables.crayon"yellow bold"
    );
    PrettyTables.pretty_table(
        permutedims(hcat(threshhold, percrejected));
        formatters    = ft_printf("%5.2f", 2:4),
        header        = header,
        header_crayon = PrettyTables.crayon"yellow bold",
        highlighters  = (hl_current, hl_column),
        tf            = tf_unicode_rounded
    )    

    # Return indexing array
    return contrast_mean .> threshhold[cont_method+1]
end



end # module


