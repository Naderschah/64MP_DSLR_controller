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




function ShiftAlign(IP::Main.Datastructures.ImagingParameters, maximum_deviation)
    """
    Almost the same as central align, but before adding the image it tries a 100x100 square of positions to see which minimizes the difference 
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