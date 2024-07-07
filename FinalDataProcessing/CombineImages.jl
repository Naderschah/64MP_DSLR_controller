# The image combination code will live here

module ImageCombining

include("Datastructures.jl")
import .Datastructures
# For MKR
include("ScalingFunctions.jl")
import .ScalingFunctions
import Images

function CentralAlign(IP::Main.Datastructures.ImagingParameters,offsets)

    # Get grid of images
    img_name_grid, img_pos_grid = GetImageGrid(IP)

    img_pos_grid = round.(img_pos_grid) .+ 200

    # Generate final array
    f_array = GenerateFinalArray(IP, img_name_grid)
    println(size(f_array))
    
    mask_array = zeros(size(f_array))

    # Midway histogram equalization reference image
    ref_img = Images.load("$(IP.path)$(img_name_grid[trunc(Int, size(img_name_grid)[1]/2), trunc(Int, size(img_name_grid)[2])])")
    # Iterate over images
    for i= 1:size(img_name_grid)[1]
        for j= 1:size(img_name_grid)[2]
            println("Loading $(IP.path)$(img_name_grid[i,j])")

            img = Images.load("$(IP.path)$(img_name_grid[i,j])")

            # Adjust histogram
            img = Images.adjust_histogram([ref_img, img], Images.MidwayEqualization(nbins = 256))[2]
            # Format for combination
            img = permutedims(Images.channelview(img), (2,3,1))
            # Orient correctly
            img = img[end:-1:1, 1:end, 1:end] # TODO Has to go
            # Offset
            off = trunc.(Int, IP.overlap .* size(img) .*2 )
            # Crop and combine              ORIGINAL
            #img = img[1+off[1]:end-off[1], 1+off[2]:end-off[2], 1:end]

            #f_x = trunc(Int, img_pos_grid[i,j,1])# inside used to be j-1 and i -1
            #f_y = trunc(Int, img_pos_grid[i,j,2])# + (j-1)*secondary_offsets[1])

            #f_array[f_x+1:f_x+size(img)[1], f_y+1:f_y+size(img)[2], 1:end] .+= img[1:end, 1:end, 1:end]

            # TRYING WITH OPACITY
            f_x = trunc(Int, img_pos_grid[i,j,1]+(j-1)*off[1])+1000
            f_y = trunc(Int, img_pos_grid[i,j,2]+(i-1)*off[2])+1000

            f_array[f_x+1:f_x+size(img)[1], f_y+1:f_y+size(img)[2], :] .+= img[1:end, 1:end, :]
            mask_array[f_x+1:f_x+size(img)[1], f_y+1:f_y+size(img)[2], :] .+= 1
        end
    end
    
    # Remove all rows and columns of just black pixels
    f_array= f_array ./ mask_array
    replace!(f_array, NaN => 0)

    f_array= permutedims(f_array, (3,1,2))
    f_array = Images.colorview(Images.RGB,f_array)

    #f_array = Images.colorview(Images.RGB, map(Images.N0f8, replace!(f_array./maximum(f_array,dims=3), NaN => 0)))
    #print(f_array[1,5:10])
    idx1 = [any(row.!=Images.RGB{Images.N0f8}(0,0,0)) for row in eachrow(f_array)]
    f_array=f_array[idx1, :]
    idx2 = [any(col.!=Images.RGB{Images.N0f8}(0,0,0)) for col in eachcol(f_array)]
    f_array = f_array[:,idx2]
    print(f_array[1,5:10])

    #mask_array=mask_array[idx1, :, :]
    #mask_array = mask_array[:,idx2, :]

    #f_array = Images.colorview(Images.RGB, map(Images.N0f8,replace!(Images.channelview(f_array) ./ permutedims(mask_array, (3,1,2)),NaN=>0)))
    

    # Return as JuliaImages array
    return f_array
end


function fnameFocused(y,z,e)
    # Generates focused filenames based on grid
    return "Focused_y=$(y)_z=$(z)_e=$(e).png"
end

function GetIdentifiers(ImagingParams)
    # Gets filename identifiers from path 
    files = readdir(ImagingParams.path)
    f_y_z_exp = [split(i, "_") for i in files]
    f_y_z = [ endswith(i[end], ".png") ? [i[1],parse(Int, String(split(i[2],"=")[2])), parse(Int, String(split(i[3], "=")[2]))] : ["F", -1, -1] for i in f_y_z_exp]
    y = unique([i[2] for i in f_y_z])
    z = unique([i[3] for i in f_y_z])
    y = sort(y)
    if -1 in y
        deleteat!(y, findall(x->x==-1, y))
    end
    z = sort(z)
    if -1 in z
        deleteat!(z, findall(x->x==-1, z))
    end
    return y, z
end

function GetImageGrid(ImagingParams, fnamefunc=fnameFocused)

    x,y = GetIdentifiers(ImagingParams)
    x = reverse(x)
    y = reverse(y)
    println("x: $(x)")
    println("y: $(y)")
    img_name_grid = Array{String}(undef,length(y), length(x))
    img_pos_grid = Array{Float64}(undef,length(y), length(x), 2)
    for i in eachindex(x)
        for j in eachindex(y)
            img_name_grid[end-j+1,end-i+1]  = fnamefunc(x[i],y[j],ImagingParams.exposure)
            img_pos_grid[j,i,:] = GeneratePxCoordinates([x[i],y[j]], ImagingParams)
        end
    end
    #return img_name_grid[end:-1:1,end:-1:1], img_pos_grid[end:-1:1,end:-1:1,:]
    println(img_name_grid)
    return img_name_grid, img_pos_grid
end


function GeneratePxCoordinates(pos, ImagingParams)
    #Function to determine the expected position of images
    return pos .* ImagingParams.steps_per_mm ./ (ImagingParams.px_size/ImagingParams.magnification)
end

function GenerateFinalArray(ImagingParams, img_name_grid)
    #=
    Here we generate the final array of images to be used for the final image
    =#
    printstyled("Creating final array", color= :red)
    im_width, im_height = size(img_name_grid) .+ 1
    println("im_width : $(im_width)") # TODO
    println("im_height: $(im_height)")
    f_height = 20000#.im_height * (1-ImagingParams.overlap) * (im_height+1) # TODO +1 tmp shouldnt be required
    f_width = 20000#.im_width * (1-ImagingParams.overlap) * (im_width+1) # TODO +1 tmp shouldnt be required
    f_height = trunc(Int,f_height)+10
    f_width = trunc(Int,f_width)+10
    final_array = zeros(f_height,f_width,3)
    println("final array size $(size(final_array))")
    return final_array
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