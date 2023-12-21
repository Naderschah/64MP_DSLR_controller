# Random writing for now

##### Copied Functions
# Compute Contrast
function LoG(I)
    #=
    Contrast using Lagrangian of Gaussian
    I -> Image
    grey -> greyscale projector (if requiredd)
    =#
    # Lagrangian of Gaussian 1.4σ
    LoG_kernel = [[0,1,1,2,2,2,1,1,0] [1,2,4,5,5,5,4,2,1] [1,4,5,3,0,3,5,4,1] [2,5,3,-12,-24,-12,3,5,2] [2,5,0,-24,-40,-24,0,5,2] [2,5,3,-12,-24,-12,3,5,2] [1,4,5,3,0,3,5,4,1] [1,2,4,5,5,5,4,2,1] [0,1,1,2,2,2,1,1,0] ]
    kernel_size = trunc(Int, size(LoG_kernel)[1]/2)
    return repeat(abs.(DSP.conv(LoG_kernel, mean(Float32.(I), dims=3)[:,:,1])[kernel_size+1:end-kernel_size,kernel_size+1:end-kernel_size,:]),outer=[1,1,3])
end

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

function GrabIdentifiers(image_directory)
    files = readdir(image_directory)
    x_y_z_exp = [split(i, "_") for i in files]
    x_y_z = [[parse(Int, String(i[1])),parse(Int, String(i[2])), parse(Int, String(i[3]))] for i in x_y_z_exp]
    exp = []
    try
        exp = unique([parse(Int, String(split(i[4], ".")[1])[4:end-3]) for i in x_y_z_exp])
    catch # For one exposure
        exp = ["NoIR"]
    end
    x = unique([i[1] for i in x_y_z])
    y = unique([i[2] for i in x_y_z])
    z = unique([i[3] for i in x_y_z])
    # We will only do one yz pos for starters
    x = sort(x)
    y = sort(y)
    z = sort(z)
    
    return x,y,z,exp
end

function GenerateFileName(x,y,z,exp)
    if typeof(exp) == Int
        return "$(x)_$(y)_$(z)_exp$(exp)mus.dng"
    else
        return "$(x)_$(y)_$(z)_$(exp).dng"
    end
end


function GeneratePxCoordinates(pos, steps_per_mm=0.00012397, px_size=1.55*10^-3, magnification=2)
    #Function to determine the expected position of images
    return pos .* steps_per_mm ./ (px_size/magnification)
end


##### New Functions

# Generate contrast data cube
function GenerateFocusCube(fnames)

    Data_cube = Array{Float32}(undef, 3040, 4056, length(fnames))
    for i in eachindex(fnames)
        img  = LoadDNGLibRaw(fnames[i])
        Data_cube[i]= LoG(img)
    end
    return Data_cube
end


function GeneratexyDistFromCamera(focus_distance, px_coordinates)


image_directory = ""

x,y,z,exps = GrabIdentifiers(image_directory)
y = y[1]
z = z[1]
exps = exps[1]
fnames = [GenerateFileName(x[i],y,z,exps) for i in eachindex(x)]

Data_cube = GenerateFocusCube(fnames)
#=
We now have a data cube with the local contrast of each image
We now need to determine what the in focus threshhold is
For now use larger than mean value
=#

In_Focus = Data_cube .> mean(Data_cube)

# For the in Focus images we need to determine their position
# x corresponds to the steps between the images, we solve relative to x = 0 
# Higher x mean the subject was moved away from the camera

# We assume in focus means focus distance away from the pixel, without any account of how focus distance changes as a function of image position
depth_of_field = 12*0.001 #millimeter
focus_distance = 11.3 # millimeter
FN = 17.6   # millimeter - Field Number
magnification=2

# These two dont agree for various reasons
FOV = FN/magnification # millimeter 
px_size = 1.55*10^-3 # millimeter/px

# Lets stick to pixel size
focus_distance ./= px_size

