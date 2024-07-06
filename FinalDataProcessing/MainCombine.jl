include("Datastructures.jl")
import .Datastructures
include("CombineImages.jl")
import .ImageCombining
using Images


offsets = [0,0]
# Create data struct
for i in range(-250,250)
    IP = Datastructures.ImagingParameters("NoIR", 2, 1.55*10^-3, 0.00019628179470651555, 0.8+(i/1000), "/SaveSpot/Felt/","/SaveSpot/Felt/combined/$(0.8+(i/1000))", 16, 480, 640,offsets)
    # Do alignment
    f_im = ImageCombining.CentralAlign(IP,offsets)
    # Save image
    Images.save("$(IP.save_path)Central_Align.png",f_im)
end