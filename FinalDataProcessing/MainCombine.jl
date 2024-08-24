include("Datastructures.jl")
import .Datastructures
include("CombineImages.jl")
import .ImageCombining
using Images
include("MIST_reimplementation.jl")
import .MIST

# Compute mm per step 
rot_steps = 4096 # If half stepped otherwise 2048
motor_deg_per_step = 360/rot_steps # degrees/ per steps for full rotation
stage_mm_per_deg = 0.5/360
mm_per_step=motor_deg_per_step*stage_mm_per_deg
offsets = [0,0]
# Here we need the exact pixel size based on the sensor dimensions and the active pixel area
px_size =  [7.564 / 4072 , 5.476 / 3176 ] .*2 # 2 for binning
"""
Best results so far for px_size
    [7.564 / 4072 , 5.476 / 3176 ] .*2  = [0.0037151277013752456, 0.0034483627204030228]
"""
# Processing Parameters
IP = Datastructures.ImagingParameters(
    "32000",                    # Exposure in ms string for some reason
    2,                          # Magnification (float)
    px_size,                    # Px_size 
    mm_per_step,                # mm traveled per step of linear stage
    0.8,                        # Image overlap in x & y 
    "/SaveSpot/LensDistortion/",          # Image Path
    "/SaveSpot/LensDistortion/combined/", # Save Path
    16,                         # Bit depth
    2028,                        # Image width      TODO: Why is this the wrong way around
    1520,                        # Image height
    offsets                     # Offsets to apply in processing
    )
# Below only needed if MIST is used
MP = MIST.MISTParameters(
    30,         # Percent overlap error
    IP.overlap*100, # Estimated Overlap in x in %
    IP.overlap*100, # Estimated overlap in y 
    10          # Repeatability
    )
println("Pixel size")
println(IP.px_size)
method = "MIST" # Choice of MIST, Central
# Do alignment
start = time()
if method == "Central"
    f_im = ImageCombining.CentralAlign(IP,offsets)
elseif method == "MIST" # TODO Orientation may become problematic here
    img_name_grid = MIST.build_img_name_grid(IP.path, [0, 0, 1])
    img_name_grid = img_name_grid[:,end:-1:1]
    # TODO Test this, i vaguely recall it generating greyscale instead of color and there being
    # Some issue with the output dimensions
    f_im = MIST.Stitch(IP.path, 
               IP.save_path, 
               img_name_grid, 
               MIST.LoG_kernel, 
               x -> x[1:end, end:-1:1, 1:end], #MIST.nothing_func, # In case some extra processing is to be applied to the image loading
               MP)  #end:-1:1
else
    println("Method not recognized")
end
println("Took $(time()-start) seconds")


# Save image
Images.save("$(IP.save_path)Central_Align.png",f_im)
