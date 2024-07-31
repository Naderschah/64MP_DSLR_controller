include("Datastructures.jl")
import .Datastructures
include("CombineImages.jl")
import .ImageCombining
using Images
include("MIST_reimplementation.jl")
import .MIST

# Compute mm per step 
motor_deg_per_step = 1/64/64*360
stage_mm_per_deg = 0.5/360
mm_per_step=motor_deg_per_step*stage_mm_per_deg

offsets = [0,0]
# Processing Parameters
IP = Datastructures.ImagingParameters(
    "16000",                    # Exposure in ms string for some reason
    2,                          # Magnification (float)
    1.55*10^-3,                 # Px_size 
    mm_per_step,                # mm traveled per step of linear stage
    0.2,                        # Image overlap in x & y 
    "/SaveSpot/Tnut/",          # Image Path
    "/SaveSpot/Tnut/combined/", # Save Path
    16,                         # Bit depth
    480,                        # Image width      TODO: Why is this the wrong way around
    640,                        # Image height
    offsets                     # Offsets to apply in processing
    )
# Below only needed if MIST is used
MP = MIST.MISTParameters(
    20,         # Percent overlap error
    IP.overlap, # Estimated Overlap in x
    IP.overlap, # Estimated overlap in y 
    10          # Repeatability
    )

method = "Central" # Choice of MIST, Central
# Do alignment
start = time()
if method == "Central"
    f_im = ImageCombining.CentralAlign(IP,offsets)
elseif method == "MIST" # TODO Orientation may become problematic here
    img_name_grid = MIST.build_img_name_grid(path, [0, 0, 1]) 
    # TODO Test this, i vaguely recall it generating greyscale instead of color and there being
    # Some issue with the output dimensions
    f_im = Stitch(IP.path, 
               IP.save_path, 
               img_name_grid, 
               MIST.LoG_kernel, 
               MIST.nothing_func, # In case some extra processing is to be applied to the image loading
               MP)
else
    println("Method not recognized")
end
println("Took $(time()-start) seconds")

# Save image
Images.save("$(IP.save_path)Central_Align.png",f_im)
