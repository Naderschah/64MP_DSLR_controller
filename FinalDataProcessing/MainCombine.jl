include("Datastructures.jl")
import .Datastructures
include("CombineImages.jl")
import .ImageCombining
using Images

# Compute mm per step 
motor_deg_per_step = 1/64/64*360
stage_mm_per_deg = 0.5/360
mm_per_step=motor_deg_per_step*stage_mm_per_deg

offsets = [0,0]
# Create data struct
IP = Datastructures.ImagingParameters("16000", 2, 1.55*10^-3, mm_per_step, 0.2, "/SaveSpot/Tnut/","/SaveSpot/Tnut/combined/", 16, 480, 640,offsets)
# Do alignment
f_im = ImageCombining.CentralAlign(IP,offsets)
# Save image
Images.save("$(IP.save_path)Central_Align.png",f_im)