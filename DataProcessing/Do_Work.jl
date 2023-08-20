#=
Hosts the process for modifying the iamges to achieve a final image
=#

include("Mosaic.jl")
include("Get_Focus.jl")
using Images
using FileIO
using JLD2 # Saving data
using BenchmarkTools
using LibRaw
using InteractiveUtils
# Declare globals constant and type annotate -- supposedly makes for better speed

const THREADED = false::Bool
const make_focus = true::Bool

const work_dir = "/home/felix/rapid_storage_1/Microscope"::String
const im_path = "$work_dir/raw"::String
const im_path_jl = "$work_dir/raw_jl"::String
const MEMUSE_LIMIT = 10
const t ="_"::String # THis is for string substitution so i dont always have to put brackets but just do $t behind the last
if THREADED
    println("Julia started with $(Threads.nthreads())")
end

function DoMap(im_path, x, y, z, t, bps)
    #=
    Helper function
    Originally hoped this would clear memory as the variables woudl go out of scope
    But didnt so now using GC.gc() fro BenchmarkTools
    =#
    #println("Memuse before getimg dat $(memuse())")
    img = GetImageDataPy("$im_path/$x$t$y$t$z$t"*"NoIR.dng", bps, "No","$im_path_jl/$x$t$y$t$z$t"*"NoIR.jld2")
    #println("Memuse after getimg dat $(memuse())")
    # Early return if no data
    if img == [0]
        return nothing
    end
    map = ComputeFocusMap(img, "l-star")
    #LibRaw.close(im_)
    save_object("$work_dir/focus/$z/$y/$x.jld2", map)
    return nothing
end


function parallel_map_creation(x_list, y, z, t, bps)
#=
Parallelizes the creation of focus maps -- Doesnt work libraw has a memory leak that cant be garbage collected in threading and pycall can only call one python instance
=#
    Threads.@threads for x in eachindex(x_list)
        DoMap(im_path, x_list[x], y, z, t, bps)
        print("$im_path/$x$t$y$t$z$t"*"NoIR.dng\n") # \033[1K
        #println("Memuse after img and map $(memuse())")
        # Check how to handle gc with multiple threads
    end
    return nothing
end

function sequential_map_creation(x_list, y, z, t, bps)
    for x in x_list
        #println("Memuse before img and map $(memuse())")
        print("$im_path/$x$t$y$t$z$t"*"NoIR.dng\n") # \033[1K
        DoMap(im_path, x, y, z, t, bps)
        #println("Memuse after img and map $(memuse())")
        # Do garbage collect when 40% mem is used, at 50% it kills the process
        # But i do not know hwy the lcoal variables pertaining to do map are stored try @inline there
        if memuse() > MEMUSE_LIMIT
            GC.gc()
            println("Cleared Memory")
        end
    end
end

function GenerateFocusMaps(file_dict)
    print("Focus Stacking\n")
    for z in keys(file_dict)
        mkdir("$work_dir/focus/$z")
        for y in keys(file_dict[z])
            _time = time()
            mkdir("$work_dir/focus/$z/$y")
            #println("Memuse before loop $(memuse())")
            if THREADED
                parallel_map_creation(file_dict[z][y], y, z, t, bps)
            else
                sequential_map_creation(file_dict[z][y], y, z, t, bps)
            end
        print("\nPos set took $(time()-_time)\n")
        print("Completed z=$z y=$y\n")
        end
    end
end

# TODO: This wont work for HDR!, first ened to group by exposure and remove exposure filename extension
if !isdir("$work_dir/focus")
    print("Creating Directories\n")
    mkdir("$work_dir/focus")
    mkdir("$work_dir/focus_stacked")
    mkdir("$work_dir/depthmaps")
    mkdir("$work_dir/raw_jl")
    print("Making File list\n")
else
    print("Directories exist, assuming step already ran\nCreating file list\n")
end

file_dict = Dict()

for i in readdir(im_path)
    print("$i\n")
    x,y,z,_ = split(i, "_")
    if haskey(file_dict, z)
        if haskey(file_dict[z], y)
            push!(file_dict[z][y], x)
        else
            file_dict[z][y] = [x]
        end
    else
        file_dict[z] = Dict(y=>[x])
    end
end


# Get image bps
println("Get image bps")
const bps = GetImageBPS("$im_path/0_0_0_"*"NoIR.dng")::Int
# Now we iterate each xyz position and create the image
if make_focus 
    GenerateFocusMaps(file_dict)
else
    println("Skipping focus maps")
end

for z in keys(file_dict)
    for y in keys(file_dict[z])
        print("Stacking z=$z y=$y\n")
        MakeFocusedImage(work_dir,im_path_jl, y, z, bps) # Saves in "$work_dir/focus_stacked/$y$t$z.tiff"
    end
end

# Load with to get rgb with color in last dimension
#img = FileIO.load(path)
#img = channelview(img)
#img = permutedims(img, [2,3,1])
## Rotate with -> before mosaiking not before stacking
#img = mapslices(rotr90,img,dims=[1,2]) #TODO rotr or rotl?


# TODO the below wil probably not work but give it a try

# This will have key z, then for each z has key y with shift to next y
match_dict = Dict()

for z in sort!(collect(keys(file_dict)))
    # We do the matching in rows then in columns, we then create some column alignments and hope that works
    if !haskey(file_dict,z)
        file_dict[z] = Dict()
    end
    local img1 = nothing
    local img2 = nothing
    for y in sort!(collect(keys(file_dict[z])))
        img1 = img2
        img2 = GetImageDataPy("$work_dir/focus_stacked/$y$t$z.tiff", bps)
        if !isnothing(img1)
            match_dict[z][y1]=Generate_Offsets(mm_shift=(Int(y)-Int(y1))*0.00012397, grey_project="l-star", match_method="CCORR")
        end
        # Keep track of  y coordinate for next round
        y1 = y
    end
end

save_object("$work_dir/relative_positions_y.jld2", match_dict)

# And now for individual z

match_dict = Dict()

img1 = nothing
img2 = nothing

yloc = len(keys(file_dict)) // 5 # Test TODO

# Grab 4 y positions in order
yind = sort!(collect(keys(first(file_dict))))
ys = [keys(yind)[yloc], keys(yind)[yloc*2],keys(yind)[yloc*3],keys(yind)[yloc*4]]

for y in ys
    if !haskey(file_dict,z)
        file_dict[y] = Dict()
    end
    for z in sort!(collect(keys(file_dict)))
        img1 = img2
        img2 = GetImageDataPy("$work_dir/focus_stacked/$y$t$z.tiff")
        if !isnothing(img1)
            match_dict[y][z1]=Generate_Offsets(grey_project="l-star", match_method="CCORR", mm_shift=(Int(z)-Int(z1))*0.00012397)
        end
        # Keep track of  z coordinate for next round
        z1 = z
    end
end

save_object("$work_dir/relative_positions_z.jld2", match_dict)


MakeMosaik(pos_map = "$work_dir/relative_positions.jld2")

# TODO: This not done
#Generate_Offsets(grey_project="l-star", match_method="CCORR")
#
#MakeMosaik()
