# The image combination code will live here

module ImageCombining

include("Datastructures.jl")
import .Datastructures
include("ContrastFunctions.jl")
import .ContrastFunctions
include("ScalingFunctions.jl")
import .ScalingFunctions
include("ScalingFunctions.jl")
import .ScalingFunctions
include("MKR_functions.jl")
import .MKR_functions
include("Kernels.jl")
import .Kernels
include("./Grey_Projectors.jl")
import .GreyProjectors

using Combinatorics
using Base.Threads
import Images
using StatsBase
using ImageQualityIndexes 
using ImageFiltering
using PrettyTables
using Statistics
using Plots
using Base.Threads

function CentralAlign(IP::Main.Datastructures.ImagingParameters,offsets;method="MKR", adjust_scaling=true)

    # Get grid of images
    img_name_grid, img_pos_grid = GetImageGrid(IP)
    # Orient correctly

    img_pos_grid = round.(img_pos_grid) 
    # Generate final array
    f_array = GenerateFinalArray(IP, img_name_grid)
    if method == "average"
        mask_array = zeros(size(f_array))
    elseif method == "MKR"
        # Add a 4th dimension to hold images with a maximum of 4 overlaps
        f_array = repeat(f_array, outer=[1, 1, 1, 4])
        f_array .= 0
    end
    # Midway histogram equalization reference image
    #ref_img = Images.load("$(IP.path)$(img_name_grid[trunc(Int, size(img_name_grid)[1]/2), trunc(Int, size(img_name_grid)[2])])")
    # Iterate over images

    # How much edge space is to be considered trash: 10% of overlap
    rem = trunc.(Int,IP.overlap / 10 .* [IP.im_width, IP.im_height])
    for i= 1:size(img_name_grid)[1] # Iterates y coordinate
        for j= 1:size(img_name_grid)[2] # Iterates z coordinate
            println("Loading $(IP.path)$(img_name_grid[i,j])")
            try
                img = Images.load("$(IP.path)$(img_name_grid[i,j])")
                img = permutedims(Images.channelview(img), (2,3,1))
            catch
                println("Image $(img_name_grid[i,j]) cant  be loaded")
                img = zeros(IP.im_width, IP.im_height, 3)
            end

            # Adjust histogram
            #img = Images.adjust_histogram([ref_img, img], Images.MidwayEqualization(nbins = 256))[2]
            # Format for combination
            
            # Orient correctly
            img = img[end:-1:1, 1:end, 1:end]

            f_x = trunc(Int, img_pos_grid[i,j,1])
            f_y = trunc(Int, img_pos_grid[i,j,2])

            
            if method == "average"
                f_array[f_x+1:f_x+size(img)[1], f_y+1:f_y+size(img)[2], :] .+= img[1:end, 1:end, :]
                mask_array[f_x+1:f_x+size(img)[1], f_y+1:f_y+size(img)[2], :] .+= 1
            elseif method == "center"
                w_diff, h_diff, _ = size(img).*IP.overlap./2 # /2 Because overlap is symmetric between images
                w_diff, h_diff = round(Int, w_diff), round(Int, h_diff)
                f_array[f_x+1+w_diff:f_x+size(img)[1]-w_diff, f_y+1+h_diff:f_y+size(img)[2]-h_diff, :] .+= img[1+w_diff:end-w_diff, 1+h_diff:end-h_diff, :]
            elseif method == "MKR"
                count = 1
                # Check if area empty, assign when empty area found
                while sum(f_array[f_x+1:f_x+size(img)[1], f_y+1:f_y+size(img)[2], :, count%4 + 1] .!= 0) != 0
                    count += 1
                    if count > 5 throw(Error("Could not find unpopulated area, something is wrong")) end
                end
                
                f_array[f_x+1+rem[1]:f_x+size(img)[1]-rem[1], f_y+1+rem[2]:f_y+size(img)[2]-rem[2], :, count%4 + 1] = img[1+rem[1]:end-rem[1], 1+rem[2]:end-rem[2], :]
            end
        end
    end
    
    # Remove all rows and columns of just black pixels
    if method == "average"
        f_array= f_array ./ mask_array
        replace!(f_array, NaN => 0)
    elseif method == "MKR"
        # We want to minimize the edges messing up the derivative, while we could add padding, this will lead to problems with the mixed metrics
        # So we will simply populate the entire array with other areas data, hopefully this will favor the layers with better overlap due to the contrast weight matrix
        h, w, c, count = size(f_array)
        # TODO: This loop takes forever - for now run with 20 threads
        # Could assign to all layers right away, then if there is a second image occupying the same space we just need to check that the data isnt replicated accross channels
        Threads.@threads for i in 1:h
            for j in 1:w
                # Find array indx with data
                idx = 0
                for l in 1:count
                    if sum(f_array[i,j,:,l] .== 0) < 3  ( idx = l  )  end
                end
                # And overwrite all but l with l, provided this is not a deadpixel area (ie the entire array isnt actually populated)
                if idx != 0
                    f_array[i,j,:,[k for k in 1:count if k!=idx]] = repeat(f_array[i,j,:,idx], outer=[1, 1, 1, 3]) 
                end
            end
        end

        filter = Kernels.GaussianKernel(5,2.5)
        # And now we apply the method on the f_array with each slize along dim 3 being a seperate image
        nlev = floor(log(minimum(size(f_array))) / log(2))
        pyr, pyr_Weight, Weight_mat = MKR_functions.GenerateEmptyPyramids(size(f_array)[1],size(f_array)[2], nlev, 4)
        #Weight_mat2 = deepcopy(Weight_mat)
        # Gen weight matrix and pyramids
        for x in 1:4 
            img_pyr = MKR_functions.Laplacian_Pyramid(f_array[:,:,:,x], nlev)
            for l in (1:nlev)  @inbounds pyr[l][:,:,:,x] = img_pyr[l]  end
            Weight_mat[:,:,:,x] = ContrastFunctions.color_STD(f_array[:,:,:,x], Float32)
            #Weight_mat2[:,:,:,x] = ContrastFunctions.LoG(f_array[:,:,:,x], GreyProjectors.lstar, Float32)
        end
        # Normalize
        Weight_mat = ScalingFunctions.ScaleWeightMatrix(Weight_mat, 1e-12)
        Weight_mat2 = ScalingFunctions.ScaleWeightMatrix(Weight_mat2, 1e-12)
        

        # and one more time -> Here we can change the relative influence
        Weight_mat = 0.5* Weight_mat + 0.5* Weight_mat2
        # Make weight pyramid
        for i in 1:4
            tmp = MKR_functions.Gaussian_Pyramid(Weight_mat[:,:,:,i],filter=filter)
            for l in (1:nlev)  pyr_Weight[l][:,:,:,i] = tmp[l] end
        end
        # Make final pyr
        fin_pyr = Dict()
        Threads.@threads for l in (1:nlev)
            @inbounds fin_pyr[l] = sum(pyr_Weight[l][:,:,:,:] .* pyr[l][:,:,:,:], dims=4)[:,:,:,1]  
        end
        Weight_mat = nothing
        pyr = nothing
        # Generate final image
        f_array = MKR_functions.Reconstruct_Laplacian_Pyramid(fin_pyr, filter=filter)
        fin_pyr = nothing
    end
    
    if adjust_scaling
        f_array = (f_array .- minimum(f_array))./ (maximum(f_array)-minimum(f_array))
    end

    f_array= permutedims(f_array, (3,1,2))
    f_array = Images.colorview(Images.RGB,f_array)

    #Clear any rows and columns that are only black (due to lazy on image size generation)
    idx1 = [any(row.!=Images.RGB{Images.N0f8}(0,0,0)) for row in eachrow(f_array)]
    f_array=f_array[idx1, :]
    idx2 = [any(col.!=Images.RGB{Images.N0f8}(0,0,0)) for col in eachcol(f_array)]
    f_array = f_array[:,idx2]

    return f_array
end



function GridAlign(IP::Main.Datastructures.ImagingParameters, img_grid_path; method="crop")
    # We load the file, order is Fname, x, y
    data = []
    open(img_grid_path) do f
        for line in eachline(f)
            split_line = split(strip(line), ",")
            split_line = [split_line[1], parse(Int, split_line[3]), parse(Int, split_line[2])]
            push!(data, split_line)
        end
    end
    data = permutedims(reduce(hcat, data), (2,1))#[1:2,:]
    # Offset for julia indexing
    data[:,2:end] .+= 1
    # Generate final grid size
    x_max, y_max = maximum(data[:,2]), maximum(data[:,3])
    x_max += IP.im_width + 1
    y_max += IP.im_height + 1 
    # Make final array 
    if method == "crop"
        final_string_array = fill("", trunc(Int64, y_max),trunc(Int64, x_max))
        full_coord_array = Array{Any}(undef, size(data, 1), 5)
        # Populate starting rows
        full_coord_array[:,1:3] = data[:,1:end] 
        # Populate ending rows g
        full_coord_array[:,4:end] = data[:,2:end] .+ reshape([IP.im_width, IP.im_height], 1, 2)
        """
        any number of images may overlap in an inconsistent fashion so simple indexing is insufficient
        While we can solve for each rectangle individually
        I feel it might be easier to approach this problem row by row/ column by column
        By iterating each row we can check what images are in this row, we can make an array containing teh 
        file names of images to be placed at this pixel
        """
        # Rows 
        Threads.@threads for i in axes(final_string_array, 1)
            row_cond = (full_coord_array[:,2] .< i) .&& (full_coord_array[:, 4] .> i) 
            # Columns
            for j in axes(final_string_array, 2)
                column_cond =  (full_coord_array[:,3] .< j) .&& (full_coord_array[:, 5] .> j) 
                # Images in selection
                img_in_selection = full_coord_array[row_cond .&& column_cond, :]
                # We now have a subset of imgs we need to solve for with content [name, xmin, ymin, xmin, xmax]
                if size(img_in_selection, 1) == 1
                    # No need to solve
                    final_string_array[i,j] = img_in_selection[1]
                elseif size(img_in_selection, 1) == 0
                    # Black pixels
                    final_string_array[i,j] = ""
                else
                    # Solve using Euclidean distance
                    img_centers_x = (img_in_selection[:,2] + img_in_selection[:,4])/2
                    img_centers_y = (img_in_selection[:,3] + img_in_selection[:,5])/2
                    euclidean_distance = sqrt.((img_centers_x .- i) .^2 + (img_centers_y .- j) .^2)
                    # Pixel to be used
                    idx = argmin(euclidean_distance)
                    final_string_array[i,j] = img_in_selection[idx,1]
                end
                
            end
        end
        # Make actual final array TODO: Bit depth
        final_array = zeros(Images.RGB{Images.N0f8},trunc(Int64, y_max),trunc(Int64, x_max))
        # And now we populate 
        Threads.@threads for k in axes(data, 1)
            # We load an image at a time
            img = Images.load("$(IP.path)$(data[k,1])")
            # We need to figure out the indices of the final location
            # And the indices of the initial location
            f_indices = [[i, j] for i in 1:size(final_string_array, 1), j in 1:size(final_string_array, 2) if final_string_array[i, j] == data[k,1]]
            if length(f_indices) != 0 
            # We now have the indices where the image goes, but we still need to select where from the indices come
            # So we first shift by the img position to get the indices relative to the original image
            # Note one offset as GIMP is zero based
            o_indices = [[i[1] - data[k,2], i[2] - data[k,3]] for i in f_indices]
            # Now we have two arrays of the same shape relating original and final img indices
            # So we can populate
            f_indices = permutedims(reduce(hcat, f_indices), (2,1))
            o_indices = permutedims(reduce(hcat, o_indices), (2,1))

            # Make Cartesian indices, otherwise it indexes linear
            f_indices = CartesianIndex.(f_indices[:, 1], f_indices[:, 2])
            o_indices = CartesianIndex.(o_indices[:, 1], o_indices[:, 2])
            final_array[f_indices] = img[o_indices]
            end # if
        end
    elseif method == "MKR"
        dtype = UInt8 # TODO Dtype adjustment
        # 8 is maximum possible number of images
        final_bool_array = fill(false, size(data, 1),trunc(Int64, y_max),trunc(Int64, x_max))
        full_coord_array = Array{Any}(undef, size(data, 1), 5)
        # Populate starting rows
        full_coord_array[:,1:3] = data[:,1:end] 
        # Populate ending rows 
        full_coord_array[:,4:end] = data[:,2:end] .+ reshape([IP.im_width, IP.im_height], 1, 2)
        """
        We will approach this in the same way as the crop method, however, instead of solving for the euclidean distance
        to decide on which pixel is used, we will leave the array with a list in the lowest dimension
        We will then select all with more than one index, and apply MKR to each segment where there is multiple
        --> Implementation will be suboptimal in favor of it working as I keep fucking up indexing at the moment
        The weight matrix for each image will be, 
            std of color -> texture proxy
            euclidean distance from image center 

        """
        # We use a bool array to save some memory, so we need a mapping dict
        fname_to_idx = Dict{String, Int}()
        for (index, filename) in enumerate(data[:, 1])
            fname_to_idx[filename] = index
        end
        println("Indexing Dictionary")
        println(fname_to_idx)
        # Rows 
        Threads.@threads for i in axes(final_bool_array, 2)
            row_cond = (full_coord_array[:,2] .< i) .&& (full_coord_array[:, 4] .> i) 
            # Columns
            for j in axes(final_bool_array, 3)
                column_cond =  (full_coord_array[:,3] .< j) .&& (full_coord_array[:, 5] .> j) 
                # Images in selection
                img_in_selection = full_coord_array[row_cond .&& column_cond, :]
                # And assign them to our bool array's respective index
                for fname in img_in_selection[:,1]
                    final_bool_array[fname_to_idx[fname],i,j] = true
                end
            end
        end
        # We now have a bool array indicating which image goes where 
        # Now we load the images, and populate all areas where only 1 image is 
        img_array = zeros(dtype,size(data,1),trunc(Int64, IP.im_width),trunc(Int64, IP.im_height), 3)
        final_array = zeros(dtype,trunc(Int64, y_max),trunc(Int64, x_max), 3)
        for i in data[:,1]
            # Load as actual values
            img_array[fname_to_idx[i],:,:,:] = reinterpret.(dtype,permutedims(Images.channelview(Images.load(joinpath(IP.path,i)))[1:3,:,:], (2,3,1))) 
        end
        # We select all portions where there is only 1 true along the last dimension
        # FIrst we filter for pixels that have only one true
        sum_array = sum(final_bool_array, dims=1)
        # I assume this always occurs but just to be sure as i dont want to debug this more
        if size(sum_array, 1) == 1
            sum_array = dropdims(sum_array, dims=1)
        end
        unique_pixel_indices = findall(x -> x == 1, sum_array)
        unique_pixel_indices = CartesianIndex.(unique_pixel_indices)
        # Free memory
        sum_array = nothing
        # Now we need to find the indeces for each result along the first dimension
        image_indices = [findfirst(final_bool_array[:, idx[1], idx[2]]) for idx in unique_pixel_indices]
        # And we merge them
        unique_pixel_indices = [(cart_idx, img_idx) for (cart_idx, img_idx) in zip(unique_pixel_indices, image_indices)]
        # Now we can asign all single pixels
        for i in unique_pixel_indices
            # i[2] is img idx, and i[1][1] and i[1][2] correspond to x,y
            # We index the original image by shifting its center position to the expected for just the image
            # More understandable if you check the for k in data loop in crop
            final_array[i[1][1],i[1][2],:] = img_array[i[2], i[1][1] - data[i[2],2] , i[1][2] - data[i[2],3], :] 
            # We also set the bool array value back to false to avoid any possible complications
            final_bool_array[i[2],i[1][1],i[1][2]] = false
        end # Replacing unique pixels
        #                   TODO: Remove the saving
        Images.save("$(IP.save_path)$(time()).png",final_array) #Images.colorview(permutedims(final_array, (3,1,2)) )
        println("Bool array size : Expected: Nim, mwidht, mheight")
        println(size(final_bool_array))
        """
        Now we have to select maximally large areas according to the following requirements
        - Maximal number of images included, i.e. if any image has a true in this area it has to be included
        - Secondary: Maximal width and height covered
        Then we can proceed to apply MKR and iteratively solve each rectangle
        This is implemented in a seperate function to return one bounding box at a time
        """
        # Placed in while loop, terminates when the returned area is 0,0,0,0
        while true
            idx1_min, idx1_max, idx2_min, idx2_max, imgs_with_true_in_area = FindOverlappingRegions(final_bool_array)
            # No more overlapping areas
            if idx1_min == -1 && idx2_min == -1 && idx1_max == -1 && idx2_max == -1
                break
                # If any are -1 throw error as this will probably cause an infinite loop
            elseif idx1_min == -1 || idx2_min == -1 || idx1_max == -1 || idx2_max == -1
                println("WARNING: One or more of the indices was -1, an error certainly occured calling continue now, but this will probably be an infinite loop")
                println("Vals found:")
                println("idx1_min, idx1_max, idx2_min, idx2_max, imgs_with_true_in_area")
                println("$idx1_min, $idx1_max, $idx2_min, $idx2_max, $imgs_with_true_in_area")
                continue
            else
                #       COMPUTING EUCLIDEAN DISTANCE FOR EACH IMAGE
                # We want to use the center distance as a weight, so we need to compute the distance for the overlapping areas
                # Quick Refresher:
                # data contains N lines with content [fname, xmin, ymin, xmax, ymax]
                midpointx = (full_coord_array[imgs_with_true_in_area, 2] + full_coord_array[imgs_with_true_in_area, 4])/2
                midpointy = (full_coord_array[imgs_with_true_in_area, 3] + full_coord_array[imgs_with_true_in_area, 5])/2
                # For the euclidean distance we simply generate an array, compute the euclidean distance
                # And then asign the slices
                height, width = size(img_array,2),size(img_array,3)
                dist_matrix = Array{Float64}(undef,height, width)
                center_x = trunc(Int, width / 2)
                center_y = trunc(Int, height / 2)
                for y in 1:height
                    for x in 1:width
                        dist_matrix[y, x] = sqrt((x - center_x)^2 + (y - center_y)^2)
                    end
                end
                # Make the array to assign slices to
                euclidean_distance = Array{Float64}(undef,length(imgs_with_true_in_area),idx1_max-idx1_min+1, idx2_max-idx2_min+1)
                ctr = 1 # Use of ctr due to i needing to be ordered according to imgs in area but euclidean_distance not sharing this order
                for i in imgs_with_true_in_area
                    euclidean_distance[ctr,:,:] = dist_matrix[idx1_min-data[i,2]:idx1_max-data[i,2], idx2_min-data[i,3]:idx2_max-data[i,3]]
                    ctr += 1
                end
                # We need to create the image subset, this is since each image has a different offset
                img_subset = Array{UInt8}(undef,length(imgs_with_true_in_area),idx1_max-idx1_min+1, idx2_max-idx2_min+1, 3)
                ctr = 1
                for i in imgs_with_true_in_area
                    # We subtract the offset from each index
                    img_subset[ctr,:,:,:] = img_array[i, idx1_min-data[i,2]:idx1_max-data[i,2], idx2_min-data[i,3]:idx2_max-data[i,3], :]
                    ctr += 1
                end
                #               DO MKR FOR SET
                # And finally we know for what images and what areas we can apply MKR
                mkr_res = DoMKRforFocusedImages(img_subset, euclidean_distance)
                # Truncate to UInt8 and rescale
                final_array[idx1_min:idx1_max, idx2_min:idx2_max, :] = trunc.(dtype, mkr_res .* 255)
                # Set completed indices to false 
                final_bool_array[imgs_with_true_in_area, idx1_min:idx1_max, idx2_min:idx2_max] .= false
                println("Completed MKR for $imgs_with_true_in_area, $idx1_min:$idx1_max, $idx2_min:$idx2_max")
                #                   TODO: Remove the saving
                Images.save("$(IP.save_path)$(time()).png",final_array)
                #                   REPEAT
            end # indx conditional checking

        end # While loop
        if any(final_bool_array)
            println("WARNING: After completion some true's are left, something went horribly wrong, good luck debugging")
        end # Last sanity check
    end # method
    # Bring it back to Images standard ordering opposite op: permutedims(Images.channelview(Images.load(final_array[x])), (2,3,1)) 
    #final_array = Images.colorview(permutedims(final_array, (3,1,2)) )
    # And done
    return final_array
end

function FindOverlappingRegions(bool_array)
    """
    bool_array : A boolean array of dimensions Nimage, widht, height

    This function will querry for rectangles of true shared between all images in this region 
        the following priorities are set:
        - First : No false value may be within the selected area's
        - Second : The maximal number of images must be included
        - Third : The maximal area in width and height must be included
    """
    layers, rows, cols = size(bool_array)
    # Named tuple to hold our return value
    best_overlap = (count = 0,  # N pixel
                    indices =[], # Img layers
                    bounds=CartesianIndex[]) # Img bounds in width and height
    
    # We check for pairs of layers with the same overlapping area (this will be most regions)
    for layer_combination in Combinatorics.combinations(1:layers, 2)
        # Take the sum for the combination 
        overlap_array = all(bool_array[layer_combination, :, :], dims=1)
        if size(overlap_array, 1) == 1
            overlap_array = dropdims(overlap_array, dims=1)
        end
        # Get all indices where we have a true value for both 
        overlap_indices = findall(x -> x, overlap_array)

        # Next combination if this combination does not share any pixels
        if isempty(overlap_indices)
            continue
        end

        # We now need to find the largest possible area that is completely true
        # We can achieve this by iterating the array pixels 
        idx1_min, idx1_max, idx2_min, idx2_max = FindFirstRectangle(overlap_array)
        # the mins always have to get populated, the max's mustnt
        # all are initialized to -1 so if -1 is returned for the mins, no area was found
        if idx1_min == -1 && idx2_min == -1 
            # No region left -> short circuit
            continue
        elseif idx1_min == -1 && idx2_min != -1 
            println("WARNING: indx1 was never triggered but 2 was")
        elseif idx1_min != -1 && idx2_min == -1 
            println("WARNING: indx2 was never triggered but 1 was")
        end
        # if none were triggered all is in order

        # We now check if any other images have true's in this area
        imgs_with_true_in_area = []
        for i in axes(bool_array, 1)
            if any(bool_array[i, idx1_min:idx1_max, idx2_min:idx2_max])
                # Append idx if array has true here
                push!(imgs_with_true_in_area, i)
            end
        end
        # in case something is wrong just try the next and print a warning to avoid instant termination
        if length(imgs_with_true_in_area) == 0 
            println("WARNING: This should never be triggered something went wrong")
            continue
        end

        overlap_array = all(bool_array[imgs_with_true_in_area, :, :], dims=1)
        if size(overlap_array, 1) == 1
            overlap_array = dropdims(overlap_array, dims=1)
        end
        # And we only have to run one more time and we have the maximal number of images with maximal area
        idx1_min, idx1_max, idx2_min, idx2_max = FindFirstRectangle(overlap_array)
        # the mins always have to get populated, the max's mustnt
        # all are initialized to -1 so if -1 is returned for the mins, no area was found
        if idx1_min == -1 && idx2_min == -1 
            # No region left -> short circuit
            continue
        elseif idx1_min == -1 && idx2_min != -1 
            println("WARNING: indx1 was never triggered but 2 was")
        elseif idx1_min != -1 && idx2_min == -1 
            println("WARNING: indx2 was never triggered but 1 was")
        end
        # if none were triggered all is in order

        # And return our indices
        return idx1_min, idx1_max, idx2_min, idx2_max, imgs_with_true_in_area

    end # layer combinations of 2 
    # Return -1 to indicate all areas have been found
    return -1,-1,-1,-1, [-1]
end # function

function FindFirstRectangle(grid)
    """
    Adaptation of "Largest Rectangle in Histogram"
    """
    rows, cols = size(grid)
    
    # Height array to hold counts of consecutive 'true's in each column
    height = zeros(Int, cols)

    max_area = 0
    best_tl = (0, 0)
    best_br = (-1, -1)

    for i in 1:rows
        for j in 1:cols
            if grid[i, j]
                height[j] += 1
            else
                height[j] = 0
            end
        end

        # Stack to store indices of the columns
        stack = Int[]
        left_limit = zeros(Int, cols)
        right_limit = fill(cols + 1, cols)

        # Calculate left limits
        for j in 1:cols
            while !isempty(stack) && height[last(stack)] >= height[j]
                pop!(stack)
            end
            left_limit[j] = isempty(stack) ? 1 : last(stack) + 1
            push!(stack, j)
        end

        # Clear stack for right limits calculation
        empty!(stack)

        # Calculate right limits
        for j in cols:-1:1
            while !isempty(stack) && height[last(stack)] >= height[j]
                pop!(stack)
            end
            right_limit[j] = isempty(stack) ? cols : last(stack) - 1
            push!(stack, j)
        end

        # Determine the maximum area rectangle for this row
        for j in 1:cols
            area = height[j] * (right_limit[j] - left_limit[j] + 1)
            if area > max_area
                max_area = area
                best_tl = (i - height[j] + 1, left_limit[j])
                best_br = (i, right_limit[j])
            end
        end
    end

    if max_area > 0
        idx1_min, idx2_min = best_tl
        idx1_max, idx2_max = best_br
        return idx1_min, idx1_max, idx2_min, idx2_max
    else
        return -1,-1,-1,-1 # or return some indication that no rectangle is found
    end
end


function FindFirstRectangleOld(overlap_array)
    """
    Helper function for 
        FindOverlappingRegions(bool_array)
    Takes as input an array where one has called all(bool_array[indcs,:,:], axis=1) for any indeces
    Determines essentially the largest possible rectangle in the image
    """
    idx1_min, idx1_max, idx2_min, idx2_max = -1,-1,-1,-1
    for i in axes(overlap_array, 1) 
        if any(overlap_array[i,:]) # we find the first where any is true
            idx1_min = i # We found the first rectangle
            break
    end end 
    if idx1_min == -1
        return idx1_min, idx1_max, idx2_min, idx2_max
    end
    # We now have the starting row of interest, and we proceed to find the ending point
    for j in axes(overlap_array, 2)
        if overlap_array[idx1_min,j]
            idx2_min = j
            break
        end
    end # col loop
    if idx2_min == -1
        return idx1_min, idx1_max, idx2_min, idx2_max
    end

    # Now we find the maximum connected index where this is true
    for j in idx2_min:last(axes(overlap_array, 2))
        if !overlap_array[idx1_min,j]
            idx2_max = j - 1 # Select the previous one as this one is now false
            break
        end
    end # col loop
    # If none was found set the last possible index
    if idx2_max == 0 
        idx2_max = last(axes(overlap_array, 2))
    end

    # And we find the last area in axes 1
    for i in idx1_min:last(axes(overlap_array, 1))
        if any(.!overlap_array[i,idx2_min:idx2_max]) # we find the first where any is false
            idx1_max = i-1 # We found the end 
            break
    end end 
    # If none was found set the last possible index
    if idx1_max == 0 
        idx1_max = last(axes(overlap_array, 1))
    end

    return idx1_min, idx1_max, idx2_min, idx2_max
end

function pad3d_array(arr, pad_w, pad_h, fill)
    # Dimensions of the original array
    N, w, h = size(arr)
    
    # New dimensions after padding
    new_w = w + 2 * pad_w
    new_h = h + 2 * pad_h
    
    # Creating a new array with the same type as the original, filled with zeros
    # Adjust the element type if the array does not contain zeros by default
    padded_arr = zeros(eltype(arr), N, new_w, new_h)
    
    # Copying the original array into the center of the new padded array
    padded_arr[:, pad_w+1:end-pad_w, pad_h+1:end-pad_h] .= arr

    if fill
        #   Pad the left and right columns
        padded_arr[:, 1:pad_w, pad_h+1:end-pad_h] .= repeat(arr[:, 1:1, :], 1, pad_w, 1)
        padded_arr[:, end-pad_w+1:end, pad_h+1:end-pad_h] .= repeat(arr[:, end:end, :], 1, pad_w, 1)

        # Now pad the top and bottom using the already padded columns
        padded_arr[:, :, 1:pad_h] .= repeat(padded_arr[:, :, pad_h+1:pad_h+1], 1, 1, pad_h)
        padded_arr[:, :, end-pad_h+1:end] .= repeat(padded_arr[:, :, end-pad_h:end-pad_h], 1, 1, pad_h)
    end
    return padded_arr
end

function pad4d_array(arr, pad_w, pad_h)
    # Dimensions of the original array
    N, w, h, c = size(arr)

    # New dimensions after padding
    new_w = w + 2 * pad_w
    new_h = h + 2 * pad_h
    
    # Creating a new array with the same type as the original
    padded_arr = similar(arr, N, new_w, new_h, c)
    
    # Copying the original array into the center of the new padded array
    padded_arr[:, pad_w+1:end-pad_w, pad_h+1:end-pad_h, :] .= arr
    
    # Pad the left and right columns
    padded_arr[:, 1:pad_w, pad_h+1:end-pad_h, :] .= repeat(arr[:, 1:1, :, :], 1, pad_w, 1, 1)
    padded_arr[:, end-pad_w+1:end, pad_h+1:end-pad_h, :] .= repeat(arr[:, end:end, :, :], 1, pad_w, 1, 1)

    # Now pad the top and bottom using the already padded columns
    padded_arr[:, :, 1:pad_h, :] .= repeat(padded_arr[:, :, pad_h+1:pad_h+1, :], 1, 1, pad_h, 1)
    padded_arr[:, :, end-pad_h+1:end, :] .= repeat(padded_arr[:, :, end-pad_h:end-pad_h, :], 1, 1, pad_h, 1)
    
    return padded_arr
end
# TODO: There has to be a way to combine pad3d and pad4d neatly without a bunch of if statements
function DoMKRforFocusedImages(imgs, euclidean_distance)
    # Modified version of ./ImageFusion.jl/MKR() that doenst load images and uses a different weight matrix
    # We pad the array 
    padding = 6
    imgs = pad4d_array(imgs, padding, padding)
    #euclidean_distance = pad3d_array(euclidean_distance, padding,padding, false)

    N, w, h, c = size(imgs)
    # Adjust image scaling
    imgs = imgs ./255
    # Number of pyramid levels
    nlev = floor(log(min(w ,h)) / log(2))
    #                       Prealocate data structures
    pyr, pyr_Weight, Weight_mat = MKR_functions.GenerateEmptyPyramids(w,h, nlev, N)
    if any(isnan.(imgs))
        println("Nan in MKR. NaN percentage $(sum(isnan.(imgs))/length(imgs))")
        # Replace nans with 0s
        replace!(imgs, NaN=>0)
    end

    #               Populate weight matrix
    # We want euclidean distance to not be too significant a factor, so we scale it to max 0.5
    #euclidean_distance ./= maximum(euclidean_distance)*2
    # We pregenerate the color standard deviation
    clrstd = Array{Float32}(undef, size(imgs,1),size(imgs,2),size(imgs,3),size(imgs,4)) 
    for x in axes(imgs,1)
        clrstd[x,:,:,:] = ContrastFunctions.color_STD(imgs[x,:,:,:], Float32)
    end

    contrast = Array{Float32}(undef, size(imgs,1),size(imgs,2),size(imgs,3),size(imgs,4)) 
    for x in axes(imgs,1)
        contrast[x,:,:,:] = ContrastFunctions.LoG(imgs[x,:,:,:],GreyProjectors.lstar, Float32)
    end
    # Normalize
    clrstd ./= maximum(clrstd)
    contrast ./= maximum(contrast)
    # And combine the weight matrix
    for x in axes(euclidean_distance, 1) #repeat(euclidean_distance[x,:,:], outer=[1,1,3]) +
        Weight_mat[:,:,:,x] =  clrstd[x,:,:,:] + contrast[x,:,:,:]
    end
    Weight_mat = ScalingFunctions.ScaleWeightMatrix(Weight_mat, 1e-12)
    Threads.@threads for i = 1:N
        tmp = MKR_functions.Gaussian_Pyramid(Weight_mat[:,:,:,i])
        for l in (1:nlev)
            pyr_Weight[l][:,:,:,i] = tmp[l]
            if any(isnan.(pyr_Weight[l][:,:,:,i]))
                println("Nan in pyramid weight matrix level $l")
            end
        end
    end
    Weight_mat = nothing
    # Make image pyramids
    Threads.@threads for x = 1:N
        img_pyr = MKR_functions.Laplacian_Pyramid(imgs[x,:,:,:], nlev)
        # Assign to final pyramid
        for l in (1:nlev) @inbounds pyr[l][:,:,:,x] = img_pyr[l]  end
    end
    # Create final pyramid
    fin_pyr = Dict()
    Threads.@threads for l in (1:nlev)
        @inbounds fin_pyr[l] = sum(pyr_Weight[l][:,:,:,:] .* pyr[l][:,:,:,:], dims=4)[:,:,:,1]  
    end
    # Reconstruct
    res = MKR_functions.Reconstruct_Laplacian_Pyramid(fin_pyr)
    # Clamp undo pad and return
    return clamp.(res, 0, 1)[padding+1:end-padding, padding+1:end-padding, :]
end


function ShiftAlign(IP::Main.Datastructures.ImagingParameters, maximum_deviation)
    """
    Almost the same as central align, but before adding the image it tries a 100x100 square of positions to see which minimizes the difference 
    Unfeasable to be used -> Hand align is far more efficient and accurate
    """
    # Get grid of images
    img_name_grid, img_pos_grid = GetImageGrid(IP)
    # Orient correctly
    img_pos_grid = round.(img_pos_grid) 
    # Generate final array
    f_array = GenerateFinalArray(IP, img_name_grid)
    
    mask_array = zeros(size(f_array))
    # First we generate an alignment grid of dimensions widht, height, 2, 3, where the second last dim is to organize shift relative to top and shift relative to left, and in the last we have i, j, score
    img_shift_grid = zeros(size(img_pos_grid)[1], size(img_pos_grid)[2])
    # we load each image and compare to one above and one to the left assuming that the first image is perfectly located
    for i= 1:size(img_name_grid)[1]
        for j= 1:size(img_name_grid)[2]
            # Filter for first and image existance
            if (i != 1 && j != 1) && isfile("$(IP.path)$(img_name_grid[i,j])")
                # Load the image under inspection
                img = Images.load("$(IP.path)$(img_name_grid[i,j])")
                img = permutedims(Images.channelview(img), (2,3,1))
                # And orient correctly
                img = img[end:-1:1, 1:end, 1:end]
                # Load comparison images
                if i != 1
                    # Load image to the left
                    img_i = Images.load("$(IP.path)$(img_name_grid[i-1,j])")
                    img_i = permutedims(Images.channelview(img_i), (2,3,1))
                    img_i = img_i[end:-1:1, 1:end, 1:end]
                    println("Main Image")
                    println("$(IP.path)$(img_name_grid[i,j])")
                    println("Compared to")
                    println("$(IP.path)$(img_name_grid[i-1,j])")
                    img_shift_grid[i,j,1,:] = FindImageShift(img_i, img, "i", IP.overlap, 100) #img_i is left of img; takes about 
                else
                    img_shift_grid[i,j,1,:] = [0,0]
                end

                if j != 1
                    # Load image to the top
                    img_j = Images.load("$(IP.path)$(img_name_grid[i,j-1])")
                    img_j = permutedims(Images.channelview(img_j), (2,3,1))

                    offset_j = FindImageShift(img_j, img, "j", IP.overlap, 100)
                else
                    offset_j = [0,0]
                end
            end
        end
    end
end



function FindImageShift(img1, img2, index, overlap, iter_range=50)
    """
    Finds shift between img1 and img2 by iterating from -iter_range to +iter_range in both dimensions and selecting for the one with the smallest square difference
    for index == "i" : img1 is on the left img2 is on the right
    for index == "j" : img1 is on the top img2 is on the bottom

    overlap :: fraction overlap

    """
    w,h,c = size(img1)
    
    # Start
    score = Inf
    shift = (0,0)
    start = time()
    
    for i in -iter_range:iter_range#-68-10:-58#+10 # 68
        for j in -iter_range:iter_range#-9-10:1 # -9
            if index == "i"
                # Left right alignment
                overlap_w = round(Int, w * overlap)
                overlap_h = h 
                # Compute indeces for width
                if i >= 0
                    r1_x_s = w - overlap_w + 1 -  i
                    r1_x_e = w
                    r2_x_s = 1
                    r2_x_e = overlap_w - i
                else 
                    r1_x_s = w - overlap_w + 1
                    r1_x_e = w + i
                    r2_x_s = 1 - i
                    r2_x_e = overlap_w
                end
                # Compute indeces for height
                if j >= 0
                    r1_y_s = 1 + j
                    r1_y_e = h
                    r2_y_s = 1
                    r2_y_e = h - j
                else 
                    r1_y_s = 1
                    r1_y_e = h + j
                    r2_y_s = 1 - j
                    r2_y_e = h
                end

            elseif index == "j"
                # Top Bottom Alignment
                overlap_w = w
                overlap_h = round(Int, h * overlap)
                if j >= 0
                    r1_y_s = h - overlap_h + 1 -  j
                    r1_y_e = h
                    r2_y_s = 1
                    r2_y_e = overlap_h - j
                else
                    r1_y_s = h - overlap_h + 1
                    r1_y_e = h + j
                    r2_y_s = 1 - j
                    r2_y_e = overlap_h
                end
                if i >= 0
                    r1_x_s = 1 + i
                    r1_x_e = w
                    r2_x_s = 1
                    r2_x_e = w - i
                else 
                    r1_x_s = 1
                    r1_x_e = w + i
                    r2_x_s = 1 - i
                    r2_x_e = w
                end
            end
            # We now have the basic indeces, we proceed to make sure the images are the same size
            # First we compute the minimum regions size
            r_w = min(r1_x_e - r1_x_s + 1, r2_x_e - r2_x_s + 1)
            r_h = min(r1_y_e - r1_y_s + 1, r2_y_e - r2_y_s + 1)
            #println("Region Size")
            #println("($(r_w), $(r_h))")
            # We now take these to generate the new endpoints
            r1_x_e = r1_x_s + r_w - 1
            r2_x_e = r2_x_s + r_w - 1
            r1_y_e = r1_y_s + r_h - 1
            r2_y_e = r2_y_s + r_h - 1

            tmp1 = img1[r1_x_s:r1_x_e, r1_y_s:r1_y_e] 
            tmp2 = img2[r2_x_s:r2_x_e, r2_y_s:r2_y_e]

            _score = sum(tmp1 .- tmp2)/length(tmp1)
            
            if score > _score 
                shift = (i,j)
            end
        end
    end
    
    return shift
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
    return img_name_grid, img_pos_grid
end


function GeneratePxCoordinates(pos, ImagingParams)
    #Function to determine the expected position of images
    return pos .* ImagingParams.steps_per_mm ./ (ImagingParams.px_size ./ImagingParams.magnification)
end

function GenerateFinalArray(ImagingParams, img_name_grid)
    #=
    Here we generate the final array of images to be used for the final image
    =#
    # 1 in case of shifts
    im_width, im_height = size(img_name_grid) .+ 1
    f_width = ImagingParams.im_width * (im_width -  (im_width-1)*ImagingParams.overlap)+ 1000
    f_height = ImagingParams.im_height * (im_height - (im_height-1)*ImagingParams.overlap)+ 1000
    final_array = zeros(trunc(Int64, f_width),trunc(Int64, f_height),3)
    println("final array size $(size(final_array))")
    return final_array
end




# Uses predefined positional parameters
# Can be computed naivly or from the MIST algorithm
# Overlaps images based on MKR for the overlapping portions
#                   INCOMPLETE WILL BE REMOVED
function MKR_combine(IP::Main.Datastructures.ImagingParameters, source_directory, img_name_grid, global_y_pos, global_x_pos)
    # For each image corner there will ideally be maximally 4 overlapping images
    # However as the neighboring images may also be shifted into this regime there may be up to 8
    # We will ignore the data from the extra 4 if this occurs
    throw("Error: Dont use this function, there is another")
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