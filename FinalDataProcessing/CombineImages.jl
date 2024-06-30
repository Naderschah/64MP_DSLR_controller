# The image combination code will live here

module ImageCombining

include("Datastructures.jl")
import .Datastructures
# For MKR
include("ScalingFunctions.jl")
import .ScalingFunctions

function CentralAlign(IP::Main.Datastructures.ImagingParameters)

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


# Uses predefined positional parameters
# Can be computed naivly or from the MIST algorithm
# Overlaps images based on MKR for the overlapping portions
function MKR_combine(IP::Main.Datastructures.ImagingParameters, source_directory, img_name_grid, global_y_pos, global_x_pos)
    # For each image corner there will ideally be maximally 4 overlapping images
    # However as the neighboring images may also be shifted into this regime there may be up to 8
    # We will ignore the data from the extra 4 if this occurs

    # Do some scaling
    global_y_pos = Int.(round.(global_y_pos .- minimum(global_y_pos) .+ 1))
    global_x_pos = Int.(round.(global_x_pos .- minimum(global_x_pos) .+ 1))

    # Create the final image array 
    stitched_img_height = Int(maximum(global_y_pos) + IP.im_height + 1)
    stitched_img_width = Int(maximum(global_x_pos) + IP.img_width + 1)
    # Undefined so that we use less ram
    I = Array{float32}(undef, stitched_img_height, stitched_img_width, 3, 4)
    # We do also need a contrast array
    C = Array{float16}(undef, stitched_img_height, stitched_img_width, 3, 4)
    # We populate alternating so first image gets first layer second gets second then this repeats in teh first row
    # Second row gets 3 and 4 and then third row gets 1 and 2 and so on
    row_bool = true
    column_bool = true
    img_index = 0
    for row in 1:axes(img_name_grid, 1)
        row_bool != row_bool
        column_counter = 0
        for column in 1:axes(img_name_grid, 2)
            column_bool != column_bool
            # Load image
            img =1  # TODO Completed image loading function with orientation etc
            # Figure out which level the image goes to 
            if      row_bool && !column_bool    img_index = 1
            elseif  !row_bool && !column_bool   img_index = 2
            elseif  row_bool && column_bool     img_index = 3
            else                                img_index = 4
            end

            # Compute final coordinates
            f_x = global_x_pos[row, column]
            f_y = global_y_pos[row, column]
            # Assign to final array
            I[f_x+1:f_x+size(img)[1], f_y+1:f_y+size(img)[2], 1:end, img_index] .= img[1:end, 1:end, 1:end]
            # Assign to contrast TODO What metrics use here
            C[f_x+1:f_x+size(img)[1], f_y+1:f_y+size(img)[2], 1:end, img_index] .= 1 #TODO Completed contrast function
        end
    end
    # Create Laplacian pyramid (also so that we save some ram)
    I = MKR_functions.Laplacian_Pyramid(img, nlev)
    # We now have the populated array with contrasts assigned, we now normalize the contrast, keeping in mind that we reset zero values
    Ctmp = C .== 0
    C = ScalingFunctions.ScaleWeightMatrix(C)
    C[Ctmp] = 0
    Ctmp = 0

    # TODO Create empty pyramid arrays
    Threads.@threads for i = 1:axes(I, 4)
        tmp = MKR_functions.Gaussian_Pyramid(Weight_mat[:,:,:,i])
        for l in (1:nlev)
            pyr_Weight[l][:,:,:,i] = tmp[l]
        end
    end
    # Combine images

    
end



function GIST_combine

end 







end # module