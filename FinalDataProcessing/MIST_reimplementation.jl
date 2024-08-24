
#=
Directly copied from Old Data processing

Here the code from MIST is reimplemented
https://github.com/usnistgov/MIST
Most of the explanatory comments are taken as well

https://raw.githubusercontent.com/wiki/USNISTGOV/MIST/assets/mist-algorithm-documentation.pdf21


TODO Min max may be implemented wrong, matlab seems to return both val and ind but either one is not allways captured figure out why
=#
module MIST



using AbstractFFTs
using Images
using StatsBase
using JLD2
using ImageTransformations
using Rotations
using CoordinateTransformations
using ImageSegmentation

DEBUG = false

function Stitch(path, save_path, img_name_grid, preprocessing_kernel=[],extra_op=nothing_func, imaging_params=Dict())
    #=
    Runs the actual stitching algorithm
    =#
    percent_overlap_error = imaging_params.percent_overlap_error
    estimated_overlap_x= imaging_params.estimated_overlap_x
    estimated_overlap_y = imaging_params.estimated_overlap_y
    repeatability = imaging_params.repeatability
    if !endswith(path,"/") path = path*"/" end
    if !isdir(save_path) mkdir(save_path) end
    # Generate grid with image names as indices
    println("Computing the PCIAM")
    if !isfile("$(save_path)pciam.jl") || !DEBUG
        # Compute translations
        y1, y2, x1, x2, cc1, cc2 = compute_PCIAM(path, img_name_grid,preprocessing_kernel,extra_op)
        save_object("$(save_path)pciam.jl", Dict("y1" => y1, "y2" => y2, "x1" => x1, "x2" => x2, "cc1" => cc1, "cc2" => cc2))

        println("Optimizing translations")
        # Correct transltions -- Currently empty function TODO
        y1, y2, x1, x2, cc1, cc2 = translation_optimization(path, img_name_grid, y1, y2, x1, x2, cc1, cc2, repeatability, percent_overlap_error , estimated_overlap_x, estimated_overlap_y, preprocessing_kernel,extra_op)
    else
        println("Using file data")
        temp = load_object("$(save_path)pciam.jl")
        y1 = temp["y1"]
        y2 = temp["y2"]
        x1 = temp["x1"]
        x2 = temp["x2"]
        cc1 = temp["cc1"]
        cc2 = temp["cc2"]
    end

    # Compute global positions
    println("Generating minimum spanning tree")
    if !isfile("$(save_path)mst.jld2") || !DEBUG
        tiling_indicator, tile_weights, global_y_img_pos, global_x_img_pos = minimum_spanning_tree(y1, y2, x1, x2, cc1, cc2)
        save_object("$(save_path)mst.jld2", Dict("tiling_indicator" => tiling_indicator, "tile_weights" => tile_weights, "global_y_img_pos" => global_y_img_pos, "global_x_img_pos" => global_x_img_pos))
    else
        println("Using file data")
        temp = load_object("$(save_path)mst.jld2")
        tiling_indicator = temp["tiling_indicator"]
        tile_weights = temp["tile_weights"]
        global_y_img_pos = temp["global_y_img_pos"]
        global_x_img_pos = temp["global_x_img_pos"]
    end
    println("Assembling final stitched image")
    # Assemble
    I = assemble_stitched_image(path, img_name_grid, global_y_img_pos, global_x_img_pos, tile_weights,"overlay", 1.5,extra_op)
    return I    
end

function lstar(I)
    conv_const = [0.212671, 0.715160, 0.072169]::Array{Float64}
    N = size(I, 1)::Int
    M = size(I, 2)::Int
    greyscale = Array{Float64}(undef, N,M)
    # We expect a 3 d image
    for i in 1:N
        for j in 1:M
            # Compute pixel wise Y :: TODO: Test if @view is more efficient here, based on docs it should be
            @inbounds greyscale[i,j] = sum(I[i,j,:].*conv_const)
            # Convert to luminance  (cond ? if true : if false)
            @inbounds greyscale[i,j] > 0.008856 ? greyscale[i,j] = 116*greyscale[i,j]^(1/3) - 16 :  greyscale[i,j] = 903.3* greyscale[i,j]
        end
    end
    return greyscale
end

function GetIdentitifiers(path)
    """Makes index of all files to be merged"""
    files = readdir(path)
    files = [i for i in files if !occursin("hist", i) & occursin(".png", i)]
    f_y_z_exp = [split(i, "_") for i in files]
    # Split all elements except for first along = and select second element
    f_y_z_exp = [[i[1], split(i[2], "=")[2], split(i[3], "=")[2], split(i[4], "=")[2]] for i in f_y_z_exp]
    x_y_e = [[parse(Int, String(i[2])),parse(Int, String(i[3])),String(split(String(i[4]), ".")[1])] for i in f_y_z_exp]
    x = unique([i[1] for i in x_y_e])
    y = unique([i[2] for i in x_y_e])
    e = unique([i[3] for i in x_y_e])
    # We will only do one yz pos for starters
    x = sort(x) 
    y = sort(y)
    e = sort(e)
    return x, y, e
end

function nothing_func(x)
    return x
end

# For image analyzing
function GetImage(path,name,kernel::AbstractArray=[],extra_op=nothing_func) 
    try
        img = load("$(path)$(name)")
        if length(kernel)>0
            fe = trunc(Int,size(kernel)[1]/2)
            img = imfilter(float32.(lstar(permutedims(channelview(img), (2,3,1)))), kernel)[fe+1:end-fe, fe+1:end-fe]
        else
            img = float32.(lstar(permutedims(channelview(img), (2,3,1))))
        end
        return extra_op(img)
    catch exp
        println("Image not found, usign zeros instead")    
        img = zeros(2020,1512)
        return extra_op(img)
    end
end

# For image analyzing with multiple kernels
function GetImage(path,name,kernel::Dict=Dict(),extra_op=nothing_func) 

    try
        img = load("$(path)$(name)")
        img = float32.(lstar(permutedims(channelview(img), (2,3,1))))  # Normal Gray projector: float32.(channelview(Gray.(img)))
        
        for i in 1:length(keys(kernel))
            fe = trunc(Int,size(kernel[i])[1]/2)
            img = imfilter(img, kernel[i])[fe+1:end-fe, fe+1:end-fe]
        end
        return extra_op(img)
    catch
        println("Image not found, usign zeros instead")    
        img = zeros(2020,1512)
        return extra_op(img)
    end

end

# For image combining
function read_img(path, name,extra_op=nothing_func)
    try
        img = load("$(path)$(name)")
        return extra_op(permutedims(channelview(img), (2,3,1))) 
    catch
        println("Image not found, usign zeros instead")    
        return zeros(2028,1520,3)
    end
end



function compute_PCIAM(path, img_grid, preprocessing_kernel=[],extra_op=nothing_func)
    """
    Computes the main component of the algorithm
    Phase Correlation Image Alignment Method
    """
    # Generate the image grid
    imgperdir = size(img_grid)
    # Generate translation matricxes
    y1 = Array{Int}(undef, imgperdir[1], imgperdir[2])
    y2 = Array{Int}(undef, imgperdir[1], imgperdir[2])
    x1= Array{Int}(undef, imgperdir[1], imgperdir[2])
    x2= Array{Int}(undef, imgperdir[1], imgperdir[2])
    cc1= Array{Float64}(undef, imgperdir[1], imgperdir[2])
    cc2 = Array{Float64}(undef, imgperdir[1], imgperdir[2])
    # Make all elements of undef arrays Nan
    y1 .= 0
    y2 .= 0
    x1 .= 0
    x2 .= 0
    cc1 .= 0
    cc2 .= 0

    # Compute grid combination pciam
    for j in 1:imgperdir[2]
        for i in 1:imgperdir[1]
            # Load the images
            img1 = GetImage(path,img_grid[i,j],preprocessing_kernel,extra_op)
            if i > 1
                img2 = GetImage(path,img_grid[i-1,j,1],preprocessing_kernel,extra_op)
                temp_y, temp_x, temp_cc = pciam(img2, img1, "Top")
                y1[i,j] = temp_y
                x1[i,j] = temp_x
                cc1[i,j] = temp_cc
            end
            if j>1 
                img2 = GetImage(path,img_grid[i,j-1,1],preprocessing_kernel,extra_op)
                temp_y, temp_x, temp_cc = pciam(img2, img1, "Left")
                y2[i,j] = temp_y
                x2[i,j] = temp_x
                cc2[i,j] = temp_cc
            end
        end
    end
    return y1, y2, x1, x2, cc1, cc2
end



function pciam(img1, img2, direction,number_of_peaks=2)
    # Phase correlation matrix (PCM)
    fc = AbstractFFTs.fft(img1) .* conj!(AbstractFFTs.fft(img2)) # Multidim fft d=2
    fc ./= abs.(fc) # Divide by magnitude
    replace!(fc, 0. => 1e-10)# Avoid zero terms
    pcm = real.(AbstractFFTs.ifft!(fc)) # Multidim ifft d=2 only keep real

    # Grab peaks from PCM
    idx =sortperm(-collect(Iterators.flatten(pcm))) # Sort descending
    idx = idx[1:number_of_peaks] # Grab top n peaks

    r,c = size(img1)

    # Compute locations of each peak (-1 since 1 based indexing)
    xy = CartesianIndices((1:r,1:c))[idx] # TODO: Removed the -1 in case it doesnt work this might be the issue
    # Peak cross correlation
    PCC = zeros(number_of_peaks, 3)
    for i in 1:number_of_peaks
        y,x,v= peak_cross_correlation(img1, img2, xy[i][1]-1, xy[i][2]-1, direction)
        PCC[i,1]=y
        PCC[i,2]=x
        PCC[i,3]=v 
    end

    ind = argmax(PCC[:,3])
    y,x,v = PCC[ind,:]
    return y,x,v
end


function subregion(img, x, y)
    h, w = size(img)

    x_st = x.+1
    x_end = x_st.-1 .+ w
    y_st = y.+1
    y_end = y_st.-1 .+ h

    x_st = max(1, min(x_st, w))
    x_end = max(1, min(x_end, w))
    y_st = max(1, min(y_st, h))
    y_end = max(1, min(y_end, h))
    # Sometimes julia thinks these are floats 
    y_end = trunc(Int, y_end)
    y_st = trunc(Int, y_st)
    x_end = trunc(Int, x_end) 
    x_st = trunc(Int, x_st)

    return img[y_st:y_end, x_st:x_end]
end

function peak_cross_correlation(Im1, Im2, x, y, direction)
    #=
    Here we narrow down the result of the fourier transfor athere are 4 possible combinations
    By construction all options are with img 1 on top and img 2 on bottom
    =#
    h, w = size(Im1)

    # Compute the real translation between the images from the possible four combinations:
    # 1) (x,y)     2) (w-x,y)       3) (x, h-y)      4) (w-x, h-y)

    # Generate the 4 possible combinations
    # The shown combinations correspond to the offset relative to  to the top left cornerof image 1
    m = [y, y,   h-y,  h-y]
    n = [x, w-x, x   , w-x]
    if direction == "Top"
        m = vcat(m,m)
        n = vcat(n,-n)
    else
        m = vcat(m,-m)
        n = vcat(n,n)
    end

    peaks = zeros(length(m),1)

    # Compute the cross correlation for each combination, the maximal value will be the correct one
    for i in eachindex(m) # TODO: This may iterate wrong
        x = n[i]
        y = m[i]
        peaks[i] = compute_crosscorrelation(subregion(Im1,x,y), subregion(Im2, -x, -y))
    end
    # The real translation values correspond to the maximum correlation between overlaping regions
    # [v, idx] = max(peaks) % tests for >= as opposed to required >
    idx = 1
    for i in 2:length(peaks)
        if peaks[i] > peaks[idx]
            idx = i
        end
    end
    v = peaks[idx]
    y = m[idx]
    x = n[idx]
    return y, x, v
end


#function compute_crosscorrelation(im1, im2)
#    # Center on mean
#    im1 = im1[:] .- mean(im1)
#    im2 = im2[:] .- mean(im2)
#
#    N = im1'*im2
#    D = sqrt(im1'*im1) * sqrt(im2'*im2)
#
#    cr = N/D
#
#    if isnan(cr) || isinf(cr)
#        cr = -1
#    end
#    return cr
#end

function compute_crosscorrelation(im1, im2)
    # Center on mean
    im1 .-= mean(im1)
    im2 .-= mean(im2)

    N = sum(im1 .* im2)
    D = sqrt(sum(im1.^2)) * sqrt(sum(im2.^2))

    cr = N/D

    if isnan(cr) || isinf(cr)
        cr = -1
    end
    return cr
end

# TODO: Compute percent overlap error somewhere
function translation_optimization(source_directory, img_name_grid, Y1, X1, Y2, X2, CC1, CC2, max_repeatability, percent_overlap_error = 5, estimated_overlap_x=50, estimated_overlap_y = 50, preprocessing_kernel=[],extra_op=nothing_func)

    # Get dimensions of an image
    nb_vertical_tiles, nb_horizontal_tiles = size(img_name_grid);
    
    tempI = read_img(source_directory, img_name_grid[1], extra_op)
    size_I= size(tempI)

    repeatability1 = 0
    repeatability2 = 0
    ConfidenceIndex1 = 0
    ConfidenceIndex2 = 0
    # correct the North translations
    try
        X1, Y1, ConfidenceIndex1, repeatability1 = correct_translation_model(X1, Y1, CC1, source_directory, img_name_grid, size_I, max_repeatability, percent_overlap_error, estimated_overlap_y, 11);
    catch err
        @warn("Translation Model Correction Failed... attempting to recover.");
        println(err)
        if isnan(max_repeatability) || isnan(estimated_overlap_y)
            throw("Unable to stitch. Set an estimated vertical overlap, estimated horizontal overlap, and repeatabilty to try again.");
        end
        repeatability1 = max_repeatability;
        ConfidenceIndex1 = zeros(size(X1));
        # replace translations with basic estimates and let hill climbing attempt to find a solution
        y = round((1-estimated_overlap_y/100)*size_I[1]);
        X1 = zeros(size(X1));
        Y1 = y.*ones(size(Y1));
        Y1[1,:] .= NaN;
        X1[1,:] .= NaN;
    end
    
    # correct the West translations
    try
        X2, Y2, ConfidenceIndex2, repeatability2 = correct_translation_model(X2, Y2, CC2, source_directory, img_name_grid, size_I, max_repeatability, percent_overlap_error, estimated_overlap_x, 22);
    catch err
        @warn("Translation Model Correction Failed... attempting to recover.");
        if isnan(max_repeatability) || isnan(estimated_overlap_y)
            throw("Unable to stitch. Set an estimated vertical overlap, estimated horizontal overlap, and repeatabilty to try again.");
        end
        repeatability2 = max_repeatability;
        ConfidenceIndex2 = zeros(size(X2));
        # replace translations with basic estimates and let hill climbing attempt to find a solution
        x = round((1-estimated_overlap_x/100)*size_I[2]);
        Y2 = zeros(size(Y2));
        X2 = x.*ones(size(X2));
        X2[:,1] .= NaN;
        Y2[:,1] .= NaN;
    end
    
    
    # repeatability search range is 2r +1 (to encompase +-r)
    r = max(repeatability1, repeatability2);
    r = 2*max(r, 1) + 1;
        
    # build the cross correlation search bounds and perform the search
    for j = 1:nb_horizontal_tiles
        # loop over the rows correcting invalid correlation values
        for i = 1:nb_vertical_tiles
            # if not the first column, and both images exist
            if j != 1 && !isempty(img_name_grid[i,j-1]) && !isempty(img_name_grid[i,j])
                bounds = [Y2[i,j]-r, Y2[i,j]+r, X2[i,j]-r, X2[i,j]+r];
                y2t, x2t, cc2t = cross_correlation_hill_climb(source_directory, img_name_grid[i,j-1], img_name_grid[i,j], bounds, X2[i,j], Y2[i,j], preprocessing_kernel,extra_op);
                Y2[i,j]  = y2t 
                X2[i,j]  =  x2t
                CC2[i,j] = cc2t
            end
            
            # if not the first row, and both images exist
            if i != 1 && !isempty(img_name_grid[i-1,j]) && !isempty(img_name_grid[i,j])
                bounds = [Y1[i,j]-r, Y1[i,j]+r, X1[i,j]-r, X1[i,j]+r];
                y1t, x1t, cc1t = cross_correlation_hill_climb(source_directory, img_name_grid[i-1,j], img_name_grid[i,j], bounds, X1[i,j], Y1[i,j], preprocessing_kernel,extra_op);
                Y1[i,j] = y1t
                X1[i,j] = x1t
                CC1[i,j] = cc1t
            end
        end
      
    end
    
    # # adjust the correlation value to reflect the confidence index, if it was a valid t (CI >= 4), give it a
    # higher weight than the other ts that had a cross correlation search performed
    ConfidenceIndex1[ConfidenceIndex1 .< 3] .= 0;
    ConfidenceIndex2[ConfidenceIndex2 .< 3] .= 0;
    CC1 .+= ConfidenceIndex1
    CC2 .+= ConfidenceIndex2
    #s = StitchingStatistics.getInstance;
    #s.global_optimization_time = toc(startTime);
    return Y1, X1, Y2, X2, CC1, CC2
end


function cross_correlation_hill_climb(images_path, I1_name, I2_name, bounds, x, y,preprocessing_kernel=[],extra_op=nothing_func)
    # TODO: Is color version better?
    I1 = GetImage(images_path, I1_name, preprocessing_kernel,extra_op)
    I2 = GetImage(images_path, I2_name, preprocessing_kernel,extra_op)

    max_peak = -1;
    
    # FIXME: If implementing the parameter group also add the first if from this function  https://github.com/usnistgov/MIST/blob/mist-matlab/src/subfunctions/cross_correlation_hill_climb.m
    if 0 > 1 # FIXME Parameter NUM_NCC_HILL_CLIMB_SEARCH_POINTS add some dict to pass this
        # perform the hill climbing from the computed translation
        # the starting point of the hill climb search is (x,y)
        x,y,max_peak = perform_hill_climb(I1, I2, bounds, x, y)

        # perform NUM_NCC_HILL_CLIMB_SEARCH_POINTS hill climbings with random starting points
        for k = 1:0 # FIXME: StitchingConstants.NUM_NCC_HILL_CLIMB_SEARCH_POINTS
            # create a new random starting point within bounds
            x1 = (bounds[4] - bounds[3])*rand() + bounds[3];
            y1 = (bounds[2] - bounds[1])*rand() + bounds[1];

            x1,y1,max_peak1 = perform_hill_climb(I1, I2, bounds, x1, y1)

            if max_peak1 > max_peak
                x = x1
                y = y1
                max_peak = max_peak1
            end
        end
    else
        # the starting point of the hill climb search is (x,y)
        x,y,max_peak = perform_hill_climb(I1, I2, bounds, x, y);
    end

    # to avoid propagating an inf value back as NCC
    if isinf(max_peak) max_peak = -1 end

    return y,x,max_peak
end 


function perform_hill_climb(I1, I2, bounds, x, y)

    max_peak = -Inf;

    # start the search at the middle point in bounds
    done = false;
    # init the matrix to hold computed ncc values to avoid recomputing
    ncc_values = zeros(3,3).=NaN

    # north, south, east, west
    dx_vals = [0;0;1;-1];
    dy_vals = [-1;1;0;0];

    # comptue the current peak
    ncc_values[2,2] = find_ncc(I1, I2, bounds, x, y);

    while !done
        # comptue the 4 connected peaks to the current locations
        for k = 1:length(dx_vals)
            delta_x = dx_vals[k]
            delta_y = dy_vals[k]
            if isnan(ncc_values[2+delta_y,2+delta_x]) # compute the NCC value if not already computed
                ncc_values[2+delta_y,2+delta_x] = find_ncc(I1, I2, bounds, x+delta_x, y+delta_y)
            end
        end
        
        
        local_max_peak,idx = maximum(ncc_values[:]), argmax(ncc_values[:])
        if isnan(local_max_peak)
            break
        end

        dxy = CartesianIndices((1:size(ncc_values)[1],1:size(ncc_values)[2]))[idx]
        delta_y,delta_x = dxy[1], dx[2]
        
        # make a translation instead of a location
        delta_y = delta_y - 2;
        delta_x = delta_x - 2;
        
        # adjust the translation value
        y = y + delta_y;
        x = x + delta_x;
        max_peak = local_max_peak;
        
        # update the elements in the ncc_values to reflect the new translation
        ncc_values = translate_mat_elements(ncc_values,delta_y,delta_x);
        # remove the 8 connected values
        ncc_values[1,1] = NaN;
        ncc_values[1,end] = NaN;
        ncc_values[end,1] = NaN;
        ncc_values[end,end] = NaN;
        
        if (delta_y == 0) && (delta_x == 0)
            done = true;
        end
    end
    return y,x,max_peak
end



function find_ncc(I1, I2, bounds, x, y)
    peak = NaN;
    
    # ensure the current location is valid
    if y < bounds[1] || y > bounds[2]
      return;
    end
    if x < bounds[3] || x > bounds[4]
      return;
    end
    
    peak = compute_crosscorrelation(subregion(I1, x, y), subregion(I2, -x, -y));

    return peak
end
    
    
function translate_mat_elements(mat,di,dj)
    if di == 0 && dj == 0
      return mat
    end
    
    m,n = size(mat) 
    
    temp = zeros(m,n) .= NaN

    for j = 1:n
      for i = 1:m
        newi = i-di;
        newj = j-dj;
        if isfinite(mat[i,j]) && newi >= 1 && newi <= m && newj >= 1 && newj <= n
            temp[newi,newj] = mat[i,j]  
        end
      end
    end
    return temp 
end





function replace_NaN_with_median(A)
    # replace NaNs with the median of the array
    A[isnan.(A)] .= median(A[!isnan.(A)]);
    return A
end


function range_filter_vec(X,Y,CC,r,V)
    #V = V .== 1 # Make a bit array for only ones this may be wrong so we using without this and see if it works

    # Compute median row values
    med_x = median(X[V])
    med_y = median(Y[V])

    # Find translation within r of the median
    valid_x_row = (X .>= (median_x-r)) && (X .<= (median_x+r))
    valid_y_row = (Y .>= (median_y-r)) && (Y .<= (median_y+r))

    # remove any translation with a CCF < 0.5
    valid_x_row = valid_x_row && (CC >= 0.5)
    valid_y_row = valid_y_row && (CC >= 0.5)

    V = valid_x_row && valid_y_row

    return V
end



function correct_translation_model(X, Y, CC, source_directory, img_name_grid, size_I, max_repeatability, percent_overlap_error, overlap, direction)

    nb_rows = size_I[1];
    nb_cols = size_I[2];
    # Initialize in function scope
    valid_translations_index = BitArray(undef, (1,1))
    # compute the estimated overlap
    if isnan(overlap)
        throw(Exception("OverlapComputationNotImplemented"))
    end

    # bound the computed image overlap (0,100)
    if overlap >= 100-percent_overlap_error
        overlap = 100-percent_overlap_error;
    end
    if overlap <= percent_overlap_error
        overlap = percent_overlap_error;
    end


    # compute range bounds
    if direction == 11
        println("North")

        ty_min = nb_rows - (overlap + percent_overlap_error)*nb_rows/100; # 
        ty_max = nb_rows - (overlap - percent_overlap_error)*nb_rows/100;

        println("ty_min $ty_min")
        println("ty_max $ty_max")
        # the valid translations are within the range bounds
        valid_translations_index = (Y.>=ty_min) .|| (Y.<=ty_max) # TODO Changed and to or here and below
        println("Sum lower $(sum(Y.>=ty_min))")
        println("Sum upper $(sum(Y.<=ty_max))")

        #s = StitchingStatistics.getInstance;
        #s.north_overlap = overlap;
        #s.north_min_range_filter = ty_min;
        #s.north_max_range_filter = ty_max;
    else
        println("Else of North - West?")
        tx_min = nb_cols - (overlap + percent_overlap_error)*nb_cols/100;
        tx_max = nb_cols - (overlap - percent_overlap_error)*nb_cols/100;

        println("tx_min $tx_min")
        println("tx_max $tx_max")
        # the valid translations are within the range bounds
        valid_translations_index = (X.>=tx_min) .|| (X.<=tx_max)
        println("Sum lower $(sum(X.>=tx_min))")
        println("Sum upper $(sum(X.<=tx_max))")

        #s = StitchingStatistics.getInstance;
        #s.west_overlap = overlap;
        #s.west_min_range_filter = tx_min;
        #s.west_max_range_filter = tx_max;
    end


    # valid translations must have a cc of >= 0.5
    println("valid_translations_index sum")
    println(sum(valid_translations_index))
    valid_translations_index[CC .< 0.7] .= 0;
    println("valid_translations_index sum post cc")
    println(sum(valid_translations_index))
    # test for existance of valid translations
    if sum(valid_translations_index.!=0) == 0
        ConfidenceIndex = zeros(size(img_name_grid));
        if direction == 22
            est_translation = round(nb_cols*(1- overlap/100));
            Y[.!isnan.(Y)] .= 0;
            X[.!isnan.(X)] .= est_translation;
        else
            est_translation = round(nb_rows*(1- overlap/100));
            X[.!isnan.(X)] .= 0;
            Y[.!isnan.(Y)] .= est_translation;
        end
    
        repeatability = 0;
        if ~isnan(max_repeatability)
            repeatability = max_repeatability;
        end

        #s = StitchingStatistics.getInstance;
        #s.north_repeatability = repeatability;

        return X, Y, ConfidenceIndex, repeatability
    end

    # filter out translation outliers
    w = 1.5; # default outlier threshold is w = 1.5

    # filter translations using outlier
    if direction == 11
    
        # filter Y components of the translations
        T = Y[valid_translations_index]
        # only filter if there are more than 3 translations
        if length(T) > 3
            q1 = median(T[T.<median(T[:])])
            q3 = median(T[T.>median(T[:])])
            iqd = abs(q3-q1);

            valid_translations_index[Y .< (q1 - w*iqd)] .= 0;
            valid_translations_index[(q3 + w*iqd) .< Y] .= 0;
        end

        # filter X components of the translations
        T = X[valid_translations_index]
        # only filter if there are more than 3 translations
        if length(T) > 3
            q1 = median(T[T.<median(T[:])])
            q3 = median(T[T.>median(T[:])])
            iqd = abs(q3-q1);

            valid_translations_index[X .< (q1 - w*iqd)] .= 0;
            valid_translations_index[(q3 + w*iqd) .< X] .= 0;
        end
    else
        
        # filter X components of the translations
        T = X[valid_translations_index]
        # only filter if there are more than 3 translations
        if length(T) > 3
            q1 = median(T[T.<median(T[:])])
            q3 = median(T[T.>median(T[:])])
            iqd = abs(q3-q1);

            valid_translations_index[X .< (q1 - w*iqd)] .= 0;
            valid_translations_index[(q3 + w*iqd) .< X] .= 0;
        end

        # filter Y components of the translations
        T = Y[valid_translations_index]
        # only filter if there are more than 3 translations
        if length(T) > 3
            q1 = median(T[T.<median(T[:])])
            q3 = median(T[T.>median(T[:])])
            iqd = abs(q3-q1);

            valid_translations_index[Y .< (q1 - w*iqd)] .= 0;
            valid_translations_index[(q3 + w*iqd) .< Y] .= 0;
        end
    end

    # test for existance of valid translations
    if sum(valid_translations_index.!=0) == 0
        ConfidenceIndex = zeros(size(img_name_grid));
        if direction == 22
            est_translation = round(nb_cols*(1- overlap/100));
            Y[!isnan.(Y)] .= 0;
            X[!isnan.(X)] .= est_translation;
        else
            est_translation = round(nb_rows*(1- overlap/100));
            X[!isnan.(X)] .= 0;
            Y[!isnan.(Y)] .= est_translation;
        end
        
        repeatability = 0;
        if !isnan(max_repeatability)
            repeatability = max_repeatability;
        end
        return X, Y, ConfidenceIndex, repeatability
    end

    # compute repeatability
    if direction == 11
        rx = ceil((maximum(X[valid_translations_index]) - minimum(X[valid_translations_index]))/2);
        tY = Y; # temporarily remove non valid translatons to compute Y range
        tY[!valid_translations_index] .= NaN;
        ry = ceil( (max(maximum(tY),2) - min(minimum(tY),2)) /2 ) # FIXME This may be wrong line 180 in source, makes no sense to me right now
        repeatability = max(rx,ry);
        
        #s = StitchingStatistics.getInstance;
        #s.north_repeatability = repeatability;

    else
        ry = ceil((maximum(Y[valid_translations_index]) - minimum(Y[valid_translations_index]))/2);
        tX = X; # temporarily remove non valid translatons to compute X range
        tX[!valid_translations_index] .= NaN;
        rx =  ceil( (max(maximum(tX),2) - min(minimum(tX),2)) /2 )     # FIXME : Same as previous
        repeatability = max(rx,ry);
        
        #s = StitchingStatistics.getInstance;
        #s.west_repeatability = repeatability;
    end

    # if the user defined a repeatabilty, use that one
    if !isnan(max_repeatability)
        repeatability = max_repeatability;
    end

    # Filter translations to ensure all are within median+-r
    if direction == 11
        for i = eachindex(axes(valid_translations_index,1))
            valid_translations_index[i,:] = range_filter_vec(X[i,:],Y[i,:],CC[i,:],repeatability,valid_translations_index[i,:]);
        end
    else
        for j = eachindex(axes(valid_translations_index,2))
            valid_translations_index[:,j] = range_filter_vec(X[:,j],Y[:,j],CC[:,j],repeatability,valid_translations_index[:,j]);
        end
    end

    # remove invalid translations
    X[!valid_translations_index] .= NaN;
    Y[!valid_translations_index] .= NaN;


    # find the rows/columns that have no valid translations
    missing_index = BitArray(undef, size(valid_translations_index)) .= 0
    if direction == 11
        # Find the rows that are missing all their values, these will later be replaced by the global median
        idxX = sum(isnan.(X),2) .== size(X,2)
        idxX[1] = 0; # remove the first row
        missing_index[idxX, :] .= 1
        
        idxY = sum(isnan.(Y),2) .== size(Y,2);
        idxY[1] = 0; # remove the first column
        missing_index[idxY, :] .= 1;
    else
        # Find the columns that are missing all their values, these will later be replaced by the global median
        idxX = sum(isnan.(X),1) == size(X,1);
        idxX[1] = 0 # remove the first val
        missing_index[:,idxX] = 1;

        idxY = sum(isnan.(Y),1) .== size(Y,1);
        idxY[1] = 0 # remove the first val
        missing_index[:,idxY] .= 1;
    end

    if direction == 11
        # fill in any invalid translations with the row wise median
        for i = eachindex(axis(X,1))
            X[i,:] = replace_NaN_with_median(X[i,:]); 
            Y[i,:] = replace_NaN_with_median(Y[i,:]);
        end
    else
        # fill in any invalid translations with the column wise median
        for j = eachindex(axis(X,2))
            X[:,j] = replace_NaN_with_median(X[:,j]);
            Y[:,j] = replace_NaN_with_median(Y[:,j]);
        end
    end


    # replace any completly missed rows by searching within the backlash using cross correlation
    if any(missing_index[:])
        X[missing_index] .= round(median(X[valid_translations_index]));
        Y[missing_index] .= round(median(Y[valid_translations_index]));
    end

    #if direction == 11  
    #    s = StitchingStatistics.getInstance;
    #    s.north_nb_tiles = nnz(~isnan(CC));
    #    s.north_nb_valid_tiles = nnz(valid_translations_index);
    #else
    #    s = StitchingStatistics.getInstance;
    #    s.west_nb_tiles = nnz(~isnan(CC));
    #    s.west_nb_valid_tiles = nnz(valid_translations_index);
    #end

    ConfidenceIndex = zeros(size(valid_translations_index));
    ConfidenceIndex[valid_translations_index] = 3 # = StitchingConstants.VALID_TRANSLATION_CC_OFFSET

    # reassert the missing tiles, to avoid propagating estimated translations for tiles that do not exist -> For mine no tiles are missing so this isnt required
    #for j = 1:size(img_name_grid,2)
    #for i = 1:size(img_name_grid,1)
    #    if isempty(img_name_grid{i,j})
    #    X[i,j] = NaN;
    #    Y[i,j] = NaN;
    #    if direction == 11
    #        if i ~= size(img_name_grid,1)
    #        # remove the translation to the tile that would be below
    #        X(i+1,j) = NaN;
    #        Y(i+1,j) = NaN;
    #        end
    #    else
    #        if j ~= size(img_name_grid,2)
    #        # remove the translation to the tile that would be to the right
    #        X(i,j+1) = NaN;
    #        Y(i,j+1) = NaN;
    #        end
    #    end
    #    end
    #end
    #end


    return X, Y, ConfidenceIndex, repeatability
end


function minimum_spanning_tree(Y1, X1, Y2, X2, CC1, CC2)

    nb_vertical_tiles, nb_horizontal_tiles = size(Y1)

    global_y_img_pos = zeros(nb_vertical_tiles,nb_horizontal_tiles)
    global_x_img_pos = zeros(nb_vertical_tiles,nb_horizontal_tiles)

    # Initialize the tiling indicator matrix that gives us the direction by which images were stitched
    # in the vertical direction up 11, down 12
    # in the horizontal direction right 21, left 22
    # This means that if an element of tiling indicator (i,j) has 11 in it, that means that this tile was stitched to the one above it (i-1,j) and
    # if an element has 21 in it that means that this tile was stitched to the one on its right (i,j+1) in the global image
    tiling_indicator = zeros(nb_vertical_tiles,nb_horizontal_tiles)

    indx1 = argmax(CC1)
    val1 = CC1[indx1]

    indx2 = argmax(CC2)
    val2 = CC2[indx2]

    if val1>val2
        ii,jj = indx1[1], indx1[2]
    else
        ii,jj = indx2[1], indx2[2]
    end

    tiling_indicator[ii,jj] = 10 # Minimum starting tile constant taken from source


    # Compute tile positions
    # Correlations are inverted since we want a maximum spannig tree
    tiling_indicator, global_y_img_pos, global_x_img_pos, tiling_coeff = minimum_spanning_tree_worker(tiling_indicator, global_y_img_pos, global_x_img_pos, Y1, X1, Y2, X2, -CC1, -CC2)
    tiling_coeff = -tiling_coeff
    tiling_coeff[ii,jj] = 5 # the starting point

    # We set 0 values to nothing 
    tiling_coeff[findall(==(0), tiling_coeff)] .= NaN
    global_x_img_pos[findall(==(0), tiling_coeff)] .= NaN
    global_y_img_pos[findall(==(0), tiling_coeff)] .= NaN
    tiling_indicator[findall(==(0), tiling_coeff)] .= NaN

    global_y_img_pos .-= minimum(global_y_img_pos) .+ 1
    global_x_img_pos .-= minimum(global_x_img_pos) .+ 1

    return tiling_indicator, tiling_coeff, global_y_img_pos, global_x_img_pos
end



function minimum_spanning_tree_worker(tiling_indicator, global_y_img_pos, global_x_img_pos, Y1, X1, Y2, X2, CC1, CC2)


    tile_y, tile_x = size(Y1)
    mst_value = 0
    tiling_coeff = zeros(tile_y, tile_x)
    tiling_coeff[tiling_indicator .> 0] .= -1

    # Keep on finding the next vertice in the tree until all is found. The first vertice is always the position of the first image
    # defined in global_y_img_pos(1,1) and global_x_img_pos(1,1)
    for j in 2:length(tiling_indicator)
        indices = findall(>(0),tiling_indicator)
        mst_min = Inf

        stitching_index, mst_i, mst_j =0, 0, 0

        # Scan all the unconnected neighbors of the connected vertices and add the one with the lowest correlation coefficient to the tree
        for i = 1:length(indices)
            # Check the neighbor below
            # Check if it is valid and isn't already stitched and that the correlation coefficient is minimal
            if indices[i][1]<tile_y && tiling_indicator[indices[i][1]+1,indices[i][2]] == 0. && CC1[indices[i][1]+1,indices[i][2]] < mst_min
                # update the minimum coefficient value
                mst_min = CC1[indices[i][1]+1,indices[i][2]]
                stitching_index = 11 # index that indicates the stitching direction of the minimal coefficient
                mst_i = indices[i][1]
                mst_j = indices[i][2]
            end

            # Check the neighbor above
            # Check if it is valid and isn't already stitched and that the correlation coefficient is minimal
            if indices[i][1]>1 && tiling_indicator[indices[i][1]-1,indices[i][2]] == 0 && CC1[indices[i][1],indices[i][2]] < mst_min
                # update the minimum coefficient value
                mst_min = CC1[indices[i][1],indices[i][2]]
                stitching_index = 12 #index that indicates the stitching direction of the minimal coefficient
                mst_i = indices[i][1]
                mst_j = indices[i][2]
            end

            # Check the neighbor to the right
            # Check if it is valid and isn't already stitched and that the correlation coefficient is minimal
            if indices[i][2]<tile_x && tiling_indicator[indices[i][1],indices[i][2]+1] == 0 && CC2[indices[i][1],indices[i][2]+1] < mst_min
                # update the minimum coefficient value
                mst_min = CC2[indices[i][1],indices[i][2]+1]
                stitching_index = 22 # index that indicates the stitching direction of the minimal coefficient
                mst_i = indices[i][1]
                mst_j = indices[i][2]
            end

            # Check the neighbor to the left
            # Check if it is valid and isn't already stitched and that the correlation coefficient is minimal
            if indices[i][2]>1 && tiling_indicator[indices[i][1],indices[i][2]-1] == 0 && CC2[indices[i][1],indices[i][2]] < mst_min
                # update the minimum coefficient value
                mst_min = CC2[indices[i][1],indices[i][2]]
                stitching_index = 21 # index that indicates the stitching direction of the minimal coefficient
                mst_i = indices[i][1]
                mst_j = indices[i][2]
            end
        end

        mst_value += mst_min
        # Compute the starting position of the chosen tile
        # Check the neighbor below
        if stitching_index == 11
            global_y_img_pos[mst_i+1,mst_j] = global_y_img_pos[mst_i,mst_j] + Y1[mst_i+1,mst_j]
            global_x_img_pos[mst_i+1,mst_j] = global_x_img_pos[mst_i,mst_j] + X1[mst_i+1,mst_j]
            # update tiling indicator
            tiling_indicator[mst_i+1,mst_j] = 11
            tiling_coeff[mst_i+1,mst_j] = mst_min
        end

        # Check the neighbor above
        if stitching_index == 12
            global_y_img_pos[mst_i-1,mst_j] = global_y_img_pos[mst_i,mst_j] - Y1[mst_i,mst_j]
            global_x_img_pos[mst_i-1,mst_j] = global_x_img_pos[mst_i,mst_j] - X1[mst_i,mst_j]
            # update tiling indicator
            tiling_indicator[mst_i-1,mst_j] = 12
            tiling_coeff[mst_i-1,mst_j] = mst_min
        end

        # Check the neighbor to the right
        if stitching_index == 22
            global_y_img_pos[mst_i,mst_j+1] = global_y_img_pos[mst_i,mst_j] + Y2[mst_i,mst_j+1]
            global_x_img_pos[mst_i,mst_j+1] = global_x_img_pos[mst_i,mst_j] + X2[mst_i,mst_j+1]
            # update tiling indicator
            tiling_indicator[mst_i,mst_j+1] = 22
            tiling_coeff[mst_i,mst_j+1] = mst_min
        end

        # Check the neighbor to the left
        if stitching_index == 21
            global_y_img_pos[mst_i,mst_j-1] = global_y_img_pos[mst_i,mst_j] - Y2[mst_i,mst_j]
            global_x_img_pos[mst_i,mst_j-1] = global_x_img_pos[mst_i,mst_j] - X2[mst_i,mst_j]
            # update tiling indicator
            tiling_indicator[mst_i,mst_j-1] = 21
            tiling_coeff[mst_i,mst_j-1] = mst_min
        end

    end
    return tiling_indicator, global_y_img_pos, global_x_img_pos, tiling_coeff
end


# TODO check if MKR can be implemented as a fusion method
function assemble_stitched_image(source_directory, img_name_grid, global_y_img_pos, global_x_img_pos, tile_weights=nothing, fusion_method="average", alpha=1.5,extra_op=nothing_func);

    if isnothing(tile_weights)
        tile_weights = ones(size(global_y_img_pos))
    end

    # Get the size of the final image
    tempI = read_img(source_directory, img_name_grid[1], extra_op)
    img_height, img_width = size(tempI)[1:2]
    tempI = 1
    # Translate position to 1,1

    global_y_img_pos = Int.(round.(global_y_img_pos .- minimum(global_y_img_pos) .+ 1))
    global_x_img_pos = Int.(round.(global_x_img_pos .- minimum(global_x_img_pos) .+ 1))
    # determine how big to make the image
    stitched_img_height = Int(maximum(global_y_img_pos) + img_height + 1)
    stitched_img_width = Int(maximum(global_x_img_pos) + img_width + 1)

    nb_img_tiles = length(img_name_grid)

    # create the ordering vector so that images with lower ccf values are placed before other images
    # the result is the images with higher ccf values overwrite those with lower values
    assemble_ordering = sortperm(collect(Iterators.flatten(tile_weights))) # TODO : THis doesnt work dims missing also output wrong
    fusion_method = lowercase(fusion_method)
    
    if fusion_method == "overlay"
        I = zeros(FixedPointNumbers.N0f16, (stitched_img_height, stitched_img_width, 3))
        # Assemble images so that the lower the image numbers get priority over higher image numbers
        # the earlier images aquired are layered upon the later images

        ### White balance testing
        # Grab histogram reference image
        ref_img = read_img(source_directory,img_name_grid[trunc(Int, size(img_name_grid)[1]/2), trunc(Int, size(img_name_grid)[2])])
        # Convert to something i can use
        ref_img=float32.(ref_img)
        ref_img .*= 255
        ref_img = trunc.(UInt8, ref_img) 
        ref_img = convert(Array{UInt8}, ref_img)
        # Do white balance
        ref_img[1,:,:]=AutoWhiteBalance(ref_img[1,:,:]);
        ref_img[2,:,:]=AutoWhiteBalance(ref_img[2,:,:]);
        ref_img[3,:,:]=AutoWhiteBalance(ref_img[3,:,:]);
        # Convert back
        ref_img = colorview(RGB,convert(Array{N0f8},permutedims(ref_img, (3,1,2))./255))
        ### End histogram



        for k = 1:nb_img_tiles
            img_idx = assemble_ordering[k] # TODO Image nme grid is it dict?
            if ~isempty(img_name_grid[img_idx])
                # Read the current image
                current_image = float32.(read_img(source_directory, img_name_grid[img_idx]))

                #### Histogram testing

                current_image .*= 255
                current_image = trunc.(UInt8, current_image) 
                current_image = convert(Array{UInt8}, current_image)
                # Do white balance
                current_image[:,:,1]=AutoWhiteBalance(current_image[:,:,1]);
                current_image[:,:,2]=AutoWhiteBalance(current_image[:,:,2]);
                current_image[:,:,3]=AutoWhiteBalance(current_image[:,:,3]);
                # Convert back
                current_image = colorview(RGB,convert(Array{N0f8},permutedims(current_image, (3,1,2))./255))

                # Do midway equalization
                current_image = permutedims(channelview(current_image), (2,3,1))

                ##### End histogram
                
                if ~isempty(current_image)
                # Assemble the image to the global one
                x_st = global_x_img_pos[img_idx]
                x_end = global_x_img_pos[img_idx]+img_width-1
                y_st = global_y_img_pos[img_idx]
                y_end = global_y_img_pos[img_idx]+img_height-1
                I[y_st:y_end,x_st:x_end,1:end] = current_image
                end
            end
        end

    elseif fusion_method == "average"
        I = zeros(Float32, (stitched_img_height, stitched_img_width, 3));
        countsI = zeros(Float32, (stitched_img_height, stitched_img_width));
        # Assemble images
        for k = 1:nb_img_tiles
            # Read the current image
            img_idx = assemble_ordering[k];
            if ~isempty(img_name_grid[img_idx])
                current_image = float32.(read_img(source_directory, img_name_grid[img_idx]))
                # Assemble the image to the global one
                x_st = global_x_img_pos[img_idx]
                x_end = global_x_img_pos[img_idx]+img_width-1
                y_st = global_y_img_pos[img_idx]
                y_end = global_y_img_pos[img_idx]+img_height-1
                I[y_st:y_end,x_st:x_end,1:end] += current_image
                countsI[y_st:y_end,x_st:x_end] .+= 1
            end
        end
        if any(isnan.(I)) println("I has nans") end
        if any(countsI.==0) 
            println("I count has $(sum(countsI.==0)) zeros") 
            #countsI[countsI .== 0] .= 1 Causes killed
            for i = eachindex(countsI)
                if countsI[i] == 0  countsI[i] = 1 end
            end
            println("Set them to 1")
        end
        I = I./countsI;
        countsI = 1
        I = FixedPointNumbers.N0f8.(I)

    elseif fusion_method == "linear"
        I = zeros{float32}(stitched_img_height, stitched_img_width);
        # generate the pixel weights matrix (its the same size as the images)
        w_mat = float32.(compute_linear_blend_pixel_weights([img_height, img_width], alpha));
        countsI = zeros{float32}(stitched_img_height, stitched_img_width);
        # Assemble images
        for k = 1:nb_img_tiles
            # Read the current image
            img_idx = assemble_ordering[k];
            if ~isempty(img_name_grid[img_idx])
                current_image = float32.(read_img(source_directory, img_name_grid[img_idx]));
                if ~isempty(current_image)
                current_image = current_image.*w_mat;
                # Assemble the image to the global one
                x_st = global_x_img_pos[img_idx];
                x_end = global_x_img_pos[img_idx]+img_width-1;
                y_st = global_y_img_pos[img_idx];
                y_end = global_y_img_pos[img_idx]+img_height-1;
                I[y_st:y_end,x_st:x_end] = I[y_st:y_end,x_st:x_end] + current_image;
                countsI[y_st:y_end,x_st:x_end] = countsI[y_st:y_end,x_st:x_end] + w_mat;
                end
            end
        end
        I = I./countsI;
        # I = cast(I,class_str); TODO

    elseif fusion_method == "min"
        I = zeros{float32}(stitched_img_height, stitched_img_width);
        maxval = typemax(float32)
        # Assemble images so that the lower the image numbers get priority over higher image numbers
        # the earlier images aquired are layered upon the later images
        for k = 1:nb_img_tiles
            img_idx = assemble_ordering[k];
            if ~isempty(img_name_grid[img_idx])
                # Read the current image
                current_image = read_img(source_directory, img_name_grid[img_idx]);
                if ~isempty(current_image)
                # Assemble the image to the global one
                x_st = global_x_img_pos[img_idx]
                x_end = global_x_img_pos[img_idx]+img_width-1
                y_st = global_y_img_pos[img_idx]
                y_end = global_y_img_pos[img_idx]+img_height-1
                temp = I[y_st:y_end,x_st:x_end]
                temp[temp .== 0] .= maxval # set the zeros to max value to avoid those being used
                I[y_st:y_end,x_st:x_end] = minimum(current_image, temp)
                end
            end
        end

    elseif fusion_method == "max"
        I = zeros(stitched_img_height, stitched_img_width, class_str);
        # Assemble images so that the lower the image numbers get priority over higher image numbers
        # the earlier images aquired are layered upon the later images
        for k = 1:nb_img_tiles
            img_idx = assemble_ordering[k];
            if ~isempty(img_name_grid[img_idx])
                # Read the current image
                current_image = read_img(source_directory, img_name_grid[img_idx]);
                if ~isempty(current_image)
                # Assemble the image to the global one
                x_st = global_x_img_pos[img_idx];
                x_end = global_x_img_pos[img_idx]+img_width-1;
                y_st = global_y_img_pos[img_idx];
                y_end = global_y_img_pos[img_idx]+img_height-1;
                I[y_st:y_end,x_st:x_end] = maximum(current_image, I[y_st:y_end,x_st:x_end]);
                end
            end
        end

    else
        # the fusion method was not valid
        error("Invalid fusion method")
    end

    return I
end

function fnamegen_varyexp(x,y,e)
   return "Focused_y=$(x)_z=$(y)_e=$(e).png"
end

function fnamegen_singleexp(x,y,e)
    return "Focused_y=$(x)_z=$(y)_NoIR.png"
end
 


function build_img_name_grid(source_img_dir,indx=[0,0,1],fnamefunc=fnamegen_varyexp)
    #=
    Here we jsut make an array of the image names to be loaded
    =#
    
    x, y, e = GetIdentitifiers(source_img_dir)
    x = reverse(x)
    println(x)
    println(y)
    println(e)
    # Reduce number of images returned by column row exp whatever
    if !isnothing(indx)
        if 0!=indx[1]
            x =[ x[indx[1]]]
        end
        if 0!=indx[2]
            y = [y[indx[2]]]
        end
        if 0!=indx[3]
            e = e[indx[3]]
        end
    end
    y = reverse(y)
    img_name_grid = Array{String}(undef,length(y), length(x))
    for i in eachindex(x)
        for j in eachindex(y)
            img_name_grid[j,i]  = fnamefunc(x[i],y[j],e)
        end
    end
    return img_name_grid#[end:-1:1,end:-1:1]
end




# White balance


function getHistogram(im_array,size_=256)
    hist = zeros(size_)
    for i in 1:size(im_array)[1]
        for j in 1:size(im_array)[2]
            hist[trunc(Int,im_array[i,j])+1] += 1
        end
    end
    return hist
end

function get_percentile(hist, percentile)
    total = sum(hist)
    current = 0
    for i in 1:length(hist)
        current += hist[i]
        if current/total >= percentile
            return i
        end
    end
    return length(hist)
end

function AutoWhiteBalance(channel, perc = 0.05,bins=256, mima=nothing)

    dtype = eltype(channel)
    if isnothing(mima)
        hist = getHistogram(channel,bins)
        mi, ma = (get_percentile(hist, perc), get_percentile(hist,100.0-perc))
    else
        mi, ma = mima
    end

    channel = clamp.((float32.(channel).-mi)./(ma-mi).*255, 0,255)

    return trunc.(dtype, channel) 
end


# Temporary struct 
struct MISTParameters
    percent_overlap_error::Int
    estimated_overlap_x::Int
    estimated_overlap_y::Int
    repeatability::Int
end

#Kernels 
emboss_kernel = [-1 -1 0; -1 0 1; 0 1 1]
sharpen_kernel = [0 -1 0; -1 5 -1; 0 -1 0]
LoG_kernel = [[0,1,1,2,2,2,1,1,0] [1,2,4,5,5,5,4,2,1] [1,4,5,3,0,3,5,4,1] [2,5,3,-12,-24,-12,3,5,2] [2,5,0,-24,-40,-24,0,5,2] [2,5,3,-12,-24,-12,3,5,2] [1,4,5,3,0,3,5,4,1] [1,2,4,5,5,5,4,2,1] [0,1,1,2,2,2,1,1,0] ]
blur_kernel = [1 2 1; 2 4 2; 1 2 1]/14



end
#
#
### Run code
#
#start = time()
## To ease identification we can run some kernels over the data
## kernels
#emboss_kernel = [-1 -1 0; -1 0 1; 0 1 1]
#sharpen_kernel = [0 -1 0; -1 5 -1; 0 -1 0]
#LoG_kernel = [[0,1,1,2,2,2,1,1,0] [1,2,4,5,5,5,4,2,1] [1,4,5,3,0,3,5,4,1] [2,5,3,-12,-24,-12,3,5,2] [2,5,0,-24,-40,-24,0,5,2] [2,5,3,-12,-24,-12,3,5,2] [1,4,5,3,0,3,5,4,1] [1,2,4,5,5,5,4,2,1] [0,1,1,2,2,2,1,1,0] ]
#blur_kernel = [1 2 1; 2 4 2; 1 2 1]/14
#preprocessing_kernel =  LoG_kernel
##preprocessing_kernel =  Dict(1 => blur_kernel, 2 => LoG_kernel)
#
#
#imaging_params = ProcessingParameters(20, 60, 60, 10)
##percent_overlap_error , estimated_overlap_x, estimated_overlap_y , repeatability 
#
#
#
#
#path = "/home/felix/rapid_storage_2/SmallWasp/"
#save_path = "/home/felix/rapid_storage_2/SmallWasp/combined/"
#
## TODO: Broccoli
## Ok so this partially fails due to the color problems, there might be some rotation in the images, but this cant solve it
## Try using offsets from central align and compute overlap from that - maybe having a better starting point makes the prediction better 
#
##img_name_grid = build_img_name_grid(path, [0, 0, 1])
#img_name_grid_orig = build_img_name_grid(path, [0, 0, 1])
## Standard orientation
##img_name_grid = img_name_grid[1:end, end:-1:1] 
##img_name_grid_orig = img_name_grid_orig[1:end, end:-1:1] 
## 2 by 2 top rigtht worked quite well so we do something
#function InvertImage(img)
#    return img[1:end, end:-1:1, :]
#end
## First attempt
#img_name_grid = img_name_grid_orig[end-3:end, 1:3]
#### Attempt stitch for normal orientation
#println(img_name_grid)
## Do stitching
## 0 -- correct orientation
#I = Stitch(path, save_path, img_name_grid, LoG_kernel, nothing_func,imaging_params)
## I = reshape(I, size(I,1), size(I,2),1) # For greyscale
#println(summary(I))
## Print dtype shape etc of I TODO feh cant open this for some reason
#save("$(save_path)MIST_combined_topleft.png", I)
#
#
#println("Took $(time()-start) seconds")
#
#
## Second Attmpt
#println(summary(img_name_grid_orig))
#siz = trunc.(Int, size(img_name_grid_orig) ./ 2)
#println(siz)
#img_name_grid = img_name_grid_orig[siz[1]-1:siz[1]+1, siz[2]-1:siz[2]+1]
#println(img_name_grid)
#### Attempt stitch for normal orientation
## Do stitching
## 0 -- correct orientation
#I = Stitch(path, save_path, img_name_grid, LoG_kernel, nothing_func,imaging_params)
## I = reshape(I, size(I,1), size(I,2),1) # For greyscale
#println(summary(I))
## Print dtype shape etc of I TODO feh cant open this for some reason
#save("$(save_path)MIST_combined_center.png", I)
#
#
#println("Took $(time()-start) seconds")

