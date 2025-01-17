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
include("./Kernels.jl")
import .Kernels
include("MKR_functions.jl")
import .MKR_functions
include("ScalingFunctions.jl")
import .ScalingFunctions

using Base.Threads
using Images
using IterTools
using ProgressMeter
using HDF5


"""
Almost the same as MainFocus, but it adhears to the realtime processing paradigm for fake HDR
and image degeneracy

The basic directory structure is now:
Images
- curr_img.txt : Contains the directory in which live processing will be handled
- img_n : The directory curr_img.txt points to, same reasoning as in the other script
    - taken_imgs : Contains the taken images
    - first_process : Contains the processed images
    - substacks : contains to be focused substacks
        - x_y_z_meta.txt : Contains information for this substack

Everything needs to be tested
"""


batch_size = 8
root_path = "/Images/"
img_nr_path = ""

"""
Flipping around open and while causes the loop to never terminate, the break is stated as being outside of a loop
and returning is unsuccessfull
TODO: Report to julia dev
"""
function read_image_path(root_path::String)
    file_path = joinpath(root_path, "curr_img.txt")
    img_nr_path = ""
    open(file_path, "r") do f
        while true
            img_nr_path = readline(f)
            # Check if img_nr_path is empty or not
            if !isempty(img_nr_path)
                break
            end
            println("Waiting for imaging to start")
            sleep(5)
        end
    end
    return img_nr_path
end

img_nr_path = read_image_path(root_path)
path = joinpath(root_path, img_nr_path)
load_path = joinpath(path,"taken_imgs")
substack_path = joinpath(path,"substacks")
save_path = joinpath(path,"first_process")
contrast_precision = Float32 
width = 2028 #3040 RAW data rn
height = 1520# 4056

debug = false

calib_data = "../CalibrationData/Calibration.hdf5"
_file = HDF5.h5open(calib_data, "r")
blackpoint = HDF5.read(_file["Blackpoint"]) 
flat = HDF5.read(_file["flat"])  # flat field
CCM = HDF5.read(_file["CCM"])  # Color correction matrix
HDF5.close(_file)

pp = Datastructures.ProcessingParameters(contrast_precision, ContrastFunctions.LoG, GreyProjectors.lstar, blackpoint, path, save_path,width, height, debug, CCM, flat)

struct Paths
    root_path::String
    path::String
    load_path::String
    substack_path::String
    save_path::String
end


paths = Paths(root_path,path, load_path, substack_path, save_path)
println(root_path)
println(path)
println(load_path)
println(substack_path)
println(save_path)


function FocusFusion(parameters::Datastructures.ProcessingParameters, paths::Paths)
    """Do not thread anything in here, risk of IO lock, only MKR is threaded"""
    # Print process ID for debugging purposes
    println("PID: $(getpid())")
    printstyled("Running with $(nthreads()) threads\n", color=:blue)
    while true
        # Read Substacks
        avail_substacs = readdir(paths.substack_path)
        if length(avail_substacs) > 0
            # Read first file
            fnames = nothing
            open(joinpath(paths.substack_path, avail_substacs[1]), "r") do f
                fnames = split(read(f, String), "\n")
            end
            # We use the meta file name as this one contains the least information irrelevant
            # to the final output
            nf_name = replace(avail_substacs[1], "_meta.txt" => "_stack.png")
            # Append path
            fnames = [joinpath(paths.load_path, i) for i in fnames]
            # Do MKR without fail check -> Just have to trust
            image = LiveProcessingMKR(fnames, parameters, 1e-12, filter=filter,apply_corrections=first)
            outpath = joinpath(save_path, nf_name)
            #TODO: Add Meta file with contrast values once timing works
            # Format for uint16 and save TODO Is png to slow?
            Images.save(outpath, trunc.(UInt16, image .* (2^16-1)))
            # Delete source images
            for i in fnames
                rm(i)
            end # Delete
            # Delete substack file
            rm(joinpath(paths.substack_path, avail_substacs[1]))
            println("Completed substack $(nf_name)")
        else
            # Check its still running
            terminate = nothing            
            open(joinpath(paths.root_path, "curr_img.txt"), "r") do f
                terminate = read(f, String) == ""
            end
            if terminate 
                break
            end
            # Sleep for 5 seconds
            println("Waiting for files")
            sleep(5)

        end # If files available
    end #While imaging flag through curr_img.txt
end# funcion


function LiveProcessingMKR(fnames, pp::Main.Datastructures.ProcessingParameters, epsilon=1e-10; raw=true, filter=MKR_functions.Pyramid_Filter(), apply_corrections=false)
    N = length(fnames)
    padding = 4
    w, h = pp.width + 2 * padding, pp.height + 2 * padding
    imgs = Array{Float32}(undef, N, w, h, 3)
    Threads.@threads for x in eachindex(fnames)
        try
            _file = HDF5.h5open(fnames[x], "r")
            img = HDF5.read(_file["image"])
            img = reinterpret(UInt16,img[1:end-16,:])
            img = MKR_functions.SimpleDebayer(img) ./ (2^12-1) # Norm to flat
            @inbounds imgs[x,:,:,:] = pad3d_array(img, padding, padding, true)
        catch e # Sometimes HDF5 error Object header/Wrong version number
            println("Failed on image $(fnames[x]) with exception:")
            println(e)
            imgs[x,:,:,:] .= 0
        end

    end
    
    nlev = floor(log(min(w, h)) / log(2))
    # Generate Empty containment objects 
    # TODO: Is this the correct way around?
    pyr, pyr_Weight, Weight_mat = MKR_functions.GenerateEmptyPyramids(w, h, nlev, N)

    # TODO Sizes correct way around?
    clrstd = Array{Float32}(undef, N, w,h,3) 
    contrast = Array{Float32}(undef, N, w,h, 3) 
    exposedness = Array{Float32}(undef, N, w,h ,3) 
    Threads.@threads for x in axes(imgs,1)
        @inbounds clrstd[x,:,:,:] = ContrastFunctions.color_STD(imgs[x,:,:,:], pp.precision)
        @inbounds greyscaled = pp.Greyscaler(imgs[x,:,:,:])
        @inbounds contrast[x,:,:,:] = pp.ContrastFunction(greyscaled, pp.precision)
        # Gaussian deviation from midway value
        @inbounds exposedness[x,:,:,:] = ContrastFunctions.WellExposedness(greyscaled ./maximum(greyscaled), sigma = 0.2)
    end
    # Normalize 
    clrstd ./= maximum(clrstd)  
    contrast ./= maximum(contrast)  
    exposedness ./= maximum(exposedness)
    # Combine 
    Threads.@threads for x in 1:N
        @inbounds Weight_mat[:,:,:,x] =  clrstd[x,:,:,:] + contrast[x,:,:,:] + exposedness[x,:,:,:] 
    end
    # Free memory
    clrstd = nothing
    contrast = nothing
    exposedness = nothing
    # Normalize
    Weight_mat = ScalingFunctions.ScaleWeightMatrix(Weight_mat, epsilon)
    # Place in pyramid
    Threads.@threads for i = 1:N
        tmp = MKR_functions.Gaussian_Pyramid(Weight_mat[:,:,:,i])
        for l in (1:nlev)
            @inbounds pyr_Weight[l][:,:,:,i] = tmp[l]
        end
    end
    # Free memory
    Weight_mat = nothing
    # And image pyramid
    Threads.@threads for x in 1:N
        img_pyr = MKR_functions.Laplacian_Pyramid(imgs[x,:,:,:], nlev)
        for l in (1:nlev) @inbounds pyr[l][:,:,:,x] = img_pyr[l]  end
    end
    # Create final pyramid
    fin_pyr = Dict()
    Threads.@threads for l in (1:nlev)
        @inbounds fin_pyr[l] = sum(pyr_Weight[l][:,:,:,:] .* pyr[l][:,:,:,:], dims=4)[:,:,:,1]  
    end
    # Reconstruct
    res = MKR_functions.Reconstruct_Laplacian_Pyramid(fin_pyr)
    # Clamp undo pad and return
    return clamp.(res, 0, 1)[padding+1:end-padding, padding+1:end-padding, :]
end


function pad3d_array(arr, pad_w, pad_h, fill)
    # Dimensions of the original array
    w, h, c = size(arr)
    
    # New dimensions after padding
    new_w = w + 2 * pad_w
    new_h = h + 2 * pad_h
    
    # Creating a new array with the same type as the original, filled with zeros
    # Adjust the element type if the array does not contain zeros by default
    padded_arr = zeros(eltype(arr), new_w, new_h, c)
    
    # Copying the original array into the center of the new padded array
    padded_arr[pad_w+1:end-pad_w, pad_h+1:end-pad_h, :] .= arr

    if fill
        #   Pad the left and right columns
        padded_arr[1:pad_w, pad_h+1:end-pad_h,:] .= repeat(arr[1:1, :, :], pad_w,1, 1)
        padded_arr[end-pad_w+1:end, pad_h+1:end-pad_h,:] .= repeat(arr[end:end,: ,:], pad_w, 1,1)

        # Now pad the top and bottom using the already padded columns
        padded_arr[:,  1:pad_h, :] .= repeat(padded_arr[:, pad_h+1:pad_h+1, :], 1, pad_h, 1)
        padded_arr[:,  end-pad_h+1:end,:] .= repeat(padded_arr[:, end-pad_h:end-pad_h,:], 1, pad_h, 1)
    end
    return padded_arr
end

FocusFusion(pp, paths)
