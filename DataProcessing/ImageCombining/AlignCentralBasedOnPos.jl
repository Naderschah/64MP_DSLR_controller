using Images
using StatsBase
# Here the images are alligned using the expected position relative to one another based on steps of the motors


struct ImagingParameters
    exposure::String
    magnification::Float64
    px_size::Float64
    steps_per_mm::Float64
    overlap::Float64
    path::String
    save_path::String
    bps::Int64
    im_width::Int64
    im_height::Int64
end







##              Expected Geometry computations

function GeneratePxCoordinates(pos, ImagingParams)
    #Function to determine the expected position of images
    return pos .* ImagingParams.steps_per_mm ./ (ImagingParams.px_size/ImagingParams.magnification)
end

function GetIdentifiers(ImagingParams)
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

function fnameFocused(x,y,e)
    #return "FeltFocused_y=$(x)_z=$(y)_e=$(e).png"
    return "FeltFocused_y=$(x)_z=$(y)_e=NoIR.png"
end

function GetImageGrid(ImagingParams,fnamefunc=fnameFocused)
    #=
    Here we jsut make an array of the image names to be loaded
    =#
    x, y = GetIdentifiers(ImagingParams)
    x = reverse(x)
    y = reverse(y)
    img_name_grid = Array{String}(undef,length(y), length(x))
    img_pos_grid = Array{Float64}(undef,length(y), length(x), 2)
    for i in eachindex(x)
        for j in eachindex(y)
            img_name_grid[j,i]  = fnamefunc(x[i],y[j],ImagingParams.exposure)
            img_pos_grid[j,i,:] = GeneratePxCoordinates([x[i],y[j]], ImagingParams)
        end
    end
    return img_name_grid[end:-1:1,end:-1:1], img_pos_grid[end:-1:1,end:-1:1,:]
end

function GenerateFinalArray(ImagingParams, img_name_grid)
    #=
    Here we generate the final array of images to be used for the final image
    =#
    im_width, im_height = size(img_name_grid) .+ 1
    
    f_height = ImagingParams.im_height * ImagingParams.overlap * (im_height+1) # TODO +1 tmp shouldnt be required
    f_width = ImagingParams.im_width * ImagingParams.overlap * (im_width+1) # TODO +1 tmp shouldnt be required
    f_height = trunc(Int,f_height)+10
    f_width = trunc(Int,f_width)+10
    final_array = zeros(f_height,f_width,3)
    return final_array
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

    channel = clamp.((float32.(channel).-mi)./(ma-mi).*(bins-1), 0,bins-1)

    return trunc.(dtype, channel) 
end









offsets = [0,0]#[274, 133]

IP = ImagingParameters("NoIR", 2, 1.55*10^-3, 0.00012397, 0.2, "/SaveSpot/Felt/","/SaveSpot/Felt/combined/", 16, 480, 640)
#IP =ImagingParameters("NoIR", 2, 1.55*10^-3, 0.00012397, 0.5, "/home/felix/rapid_storage_2/SmallWasp/", "/home/felix/rapid_storage_2/SmallWasp/combined/", 16, 4056, 3040)
# White balance spec
wb =false
wb_bins = 2^16 # If this is 256 then UInt8 is used else UInt16
# Midway equalization spec
hist = false
me_bins = 2^16
hist_spec_column = []
# Perspective is the problem

# Determine working dtypes
working_dtype = (256 >= wb_bins) ? UInt8  : UInt16
img_dtype =     (256 >= wb_bins) ? N0f8   : N0f16

img_name_grid, img_pos_grid = GetImageGrid(IP)
img_pos_grid = round.(img_pos_grid) .+ 200
println("Image grid size: $(size(img_name_grid))")
println("Image pos size: $(size(img_pos_grid))")

f_array = GenerateFinalArray(IP, img_name_grid)
println("Final array size: $(size(f_array))")

# Check all files exist
for i in eachindex(img_name_grid)
    if !isfile("$(IP.path)$(img_name_grid[i])")
        println("$(IP.path)$(img_name_grid[i]) does not exist")
    end
end
println()

# Now we start populating
length_ = length(img_name_grid)

# Image for midway hist equalization
if hist || length(hist_spec_column) > 0
    ref_img = load("$(IP.path)$(img_name_grid[trunc(Int, size(img_name_grid)[1]/2), trunc(Int, size(img_name_grid)[2])])")
end

if wb && (hist || length(hist_spec_column) > 0)
    # Convert to something i can use
    ref_img=float32.(channelview(ref_img))
    ref_img .*= (wb_bins-1)
    ref_img = trunc.(working_dtype, ref_img) 
    ref_img = convert(Array{working_dtype}, ref_img)
    # Do white balance
    ref_img[1,:,:]=AutoWhiteBalance(ref_img[1,:,:],0.05, wb_bins);
    ref_img[2,:,:]=AutoWhiteBalance(ref_img[2,:,:],0.05, wb_bins);
    ref_img[3,:,:]=AutoWhiteBalance(ref_img[3,:,:],0.05, wb_bins);
    # Convert back
    ref_img = colorview(RGB,convert(Array{img_dtype},ref_img./(wb_bins-1)))
end

for i= 1:size(img_name_grid)[1]
for j= 1:size(img_name_grid)[2]
    println("Loading $(IP.path)$(img_name_grid[i,j])")
    try
        img = load("$(IP.path)$(img_name_grid[i,j])")
        println(size(img))
        # Do white balance
        if wb
            # Convert to something i can use
            img=float32.(channelview(img))
            img .*= (wb_bins-1)
            img = trunc.(working_dtype, img) 
            img = convert(Array{working_dtype}, img)
            # Do white balance
            img[1,:,:]=AutoWhiteBalance(img[1,:,:],0.05, wb_bins);
            img[2,:,:]=AutoWhiteBalance(img[2,:,:],0.05, wb_bins);
            img[3,:,:]=AutoWhiteBalance(img[3,:,:],0.05, wb_bins);
            # Convert back
            img = colorview(RGB,convert(Array{img_dtype},img./(wb_bins-1)))
        end

        # Do histogram -- best used for specific columns with white balance applied
        if hist || (i in hist_spec_column)
            img = adjust_histogram([ref_img, img], MidwayEqualization(nbins = me_bins))[2]
        end
        img = permutedims(channelview(img), (2,3,1))
        img = img[end:-1:1, 1:end, 1:end]
        # Select central portion
        off = trunc.(Int, IP.overlap .* size(img) ./2)
        img = img[1+off[1]:end-off[1], 1+off[2]:end-off[2], 1:end]

        # Compute final coordinates
        f_x = trunc(Int, img_pos_grid[i,j,1] - (j-1)*offsets[2])# + (i-1)*secondary_offsets[2])
        f_y = trunc(Int, img_pos_grid[i,j,2] - (i-1)*offsets[1])# + (j-1)*secondary_offsets[1])

        f_array[f_x+1:f_x+size(img)[1], f_y+1:f_y+size(img)[2], 1:end] .= img[1:end, 1:end, 1:end]
    catch e
        println("   Failed to load or process $(IP.path)$(img_name_grid[i,j]) : Skipping")
        println(" Error: ",typeof(e))
        println(" Error: ",e.message)
    end
    #save("$(IP.save_path)$(i)_$j.png",f_array)
end
end

# Remove all rows and columns of just black pixels
f_array= permutedims(f_array, (3,1,2))
f_array = colorview(RGB, f_array)
f_array=f_array[[any(row.!=RGB{N0f8}(0,0,0)) for row in eachrow(f_array)], :]
f_array = f_array[:,[any(col.!=RGB{N0f8}(0,0,0)) for col in eachcol(f_array)]]

save("$(IP.save_path)Central_Align.png",f_array)