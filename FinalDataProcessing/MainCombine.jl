include("Datastructures.jl")
#import .Datastructures
include("CombineImages.jl")
import .ImageCombining



offsets = [0,0]
# Create data struct
#IP = Datastructures.ImagingParameters("NoIR", 2, 1.55*10^-3, 0.00012397, 0.2, "/SaveSpot/Felt/","/SaveSpot/Felt/combined/", 16, 480, 640,offsets)
# Do alignment
f_im = ImageCombining.CentralAlign(IP)
# Save image
save("$(IP.save_path)Central_Align.png",f_array)