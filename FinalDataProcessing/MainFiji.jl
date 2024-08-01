# Used to rename files for Fiji usage
# Later will also call Fiji directly


include("./IO_dp.jl")
import .IO_dp
include("./Datastructures.jl")
import .Datastructures
using JLD2

path = "/media/felix/3677421f-5daf-4ea7-ba33-31b09d11edcf/Images/SmallWasp/"
save_path = "/mnt/rapid_storage_2/Leaf/"
blackpoint = [0,0,0]
contrast_precision = Float32 
width = 4056
height = 3040
debug = false
pp = IO_dp.Datastructures.ProcessingParameters(contrast_precision, println, println, blackpoint, path, save_path,width, height, debug)

#img_name_grid = jldopen(save_path*"img_name_grid.jld2")["img_name_grid"]
# Runs the change filenames
img_name_grid = IO_dp.RowColumnIndexNaming(pp)

jldsave(save_path*"img_name_grid.jld2"; img_name_grid)

println("Done")