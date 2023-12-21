include("Datastructures.jl")
import .Datastructures
include("CombineImages.jl")
import .ImageCombining



offsets = [230,128]#[274, 133]
# Create data struct
IP = Datastructures.ImagingParameters("NoIR", 2, 1.55*10^-3, 0.00012397, 0.5, "/home/felix/rapid_storage_2/BananaPeal/", "/home/felix/rapid_storage_2/BananaPeal/combined/", 16, 4056, 3040,offsets)
# Do alignment
f_im = ImageCombining.CentralAlign(IP)
# Save image
save("$(IP.save_path)Central_Align.png",f_array)