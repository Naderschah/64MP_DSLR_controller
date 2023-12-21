# The image combination code will live here

module ImageCombining

include("Datastructures.jl")
import .Datastructures


function CentralAlign(IP:Datastructures.ImagingParameters)

    # Get grid of images
    img_name_grid, img_pos_grid = GetImageGrid(IP)

    img_pos_grid = round.(img_pos_grid) .+ 200

    # Generate final array
    f_array = GenerateFinalArray(IP, img_name_grid)

    # Midway histogram equalization reference image
    ref_img = load("$(IP.path)$(img_name_grid[trunc(Int, size(img_name_grid)[1]/2), trunc(Int, size(img_name_grid)[2])])")


    # Iterate over images
    for i= 1:size(img_name_grid)[1]
        for j= 1:size(img_name_grid)[2]
            println("Loading $(IP.path)$(img_name_grid[i,j])")

            img = load("$(IP.path)$(img_name_grid[i,j])")

            # Adjust histogram
            img = adjust_histogram([ref_img, img], MidwayEqualization(nbins = 256))[2]
            # Format for combination
            img = permutedims(channelview(img), (2,3,1))
            # Orient correctly
            img = img[end:-1:1, 1:end, 1:end]
            # Select central portion
            off = trunc.(Int, IP.overlap .* size(img) ./2)
            img = img[1+off[1]:end-off[1], 1+off[2]:end-off[2], 1:end]
        
            # Compute final coordinates
            f_x = trunc(Int, img_pos_grid[i,j,1] - (j-1)*offsets[2])# + (i-1)*secondary_offsets[2])
            f_y = trunc(Int, img_pos_grid[i,j,2] - (i-1)*offsets[1])# + (j-1)*secondary_offsets[1])
        
            f_array[f_x+1:f_x+size(img)[1], f_y+1:f_y+size(img)[2], 1:end] .= img[1:end, 1:end, 1:end]
        
        end
    end

    # Remove all rows and columns of just black pixels
    f_array= permutedims(f_array, (3,1,2))
    f_array = colorview(RGB, f_array)
    f_array=f_array[[any(row.!=RGB{N0f8}(0,0,0)) for row in eachrow(f_array)], :]
    f_array = f_array[:,[any(col.!=RGB{N0f8}(0,0,0)) for col in eachcol(f_array)]]

    # Return as JuliaImages array
    return f_array
end










end # module