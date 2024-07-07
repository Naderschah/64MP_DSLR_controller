include("Datastructures.jl")
import .Datastructures
include("CombineImages.jl")
import .ImageCombining
using Images


offsets = [0,0]
# Create data struct
IP = Datastructures.ImagingParameters("NoIR", 2, 1.55*10^-3, 0.00019628179470651555, 0.8, "/SaveSpot/Tnut/","/SaveSpot/Tnut/combined/", 16, 480, 640,offsets)
# Do alignment
f_im = ImageCombining.CentralAlign(IP,offsets)
# Save image
Images.save("$(IP.save_path)Central_Align.png",f_im)