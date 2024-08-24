
using FixedPointNumbers
using HDF5
using Images

function unpack_12bit_packed(data, width, height)
    # Calculate the new length of the data array after removing extra pixels
    # Assuming data is a flat 1D array of bytes
    new_width = Int((width - 28)/1.5)  # Adjust based on actual extra pixels
    data = data[1:end,:,:] # Setting starting to 15 makes a weird cool image
    # Prepare an array to hold the unpacked 16-bit pixel values
    unpacked = Array{UInt16}(undef, new_width * height)

    # Index for unpacked array
    j = 1

    # Process every 3 bytes to get 2 pixels
    for i in 1:3:length(data) - 2
        # First 12-bit pixel
        first_pixel = (UInt16(data[i]) << 4) | (UInt16(data[i + 1] & 0xF0) >> 4)
        unpacked[j] = first_pixel
        j += 1

        # Second 12-bit pixel
        second_pixel = (UInt16(data[i + 1] & 0x0F) << 8) | UInt16(data[i + 2])
        unpacked[j] = second_pixel
        j += 1

        # Break if the array is filled
        if j > new_width * height
            break
        end
    end

    # Reshape the unpacked array to the correct dimensions
    unpacked = reshape(unpacked, new_width, height)
    # And rescale to uint 16
    return  UInt16.(((unpacked ./ (4095)) .* typemax(UInt16) .÷ 1)) 
end
function SimpleDebayer(img, bayer_patter = "BGGR")
    """Reduces image size by half simply overlapping the pixels"""
    output = Array{UInt16}(undef,(Int(size(img, 1) / 2), Int(size(img, 2)/ 2), 3))
    # TODO: Generalize for differnt bayer orders, or not ltos of operations involved, and this isnt going to change
    for i in 1:2:size(img,1)-1
        for j in 1:2:size(img,2)-1
            # B (Blue)
            output[i ÷ 2 + 1, j ÷ 2 + 1, 3] = img[i, j]
            # G1 (Green)
            output[i ÷ 2 + 1, j ÷ 2 + 1, 2] = (img[i, j+1]÷2 + img[i+1, j]÷2)
            # R (Red)
            output[i ÷ 2 + 1, j ÷ 2 + 1, 1] = img[i+1, j+1]
        end
    end
    return output
end


path = "/Images/img_0/11000_0_0_exp32000.hdf5"

_file = HDF5.h5open(path, "r")

img = HDF5.read(_file["image"])
HDF5.close(_file)
print(size(img))
#img = unpack_12bit_packed(img, size(img,1), size(img,2))
img = reinterpret(UInt16,img)
print(size(img))

# Debayer
img = SimpleDebayer(img) 

println(typeof(img))
println(sum(img))
println(maximum(img)/typemax(UInt16))
println(minimum(img)/typemax(UInt16))
println(size(img))
println(sum(img)/length(img)/typemax(UInt16))
# Now save and check
img = colorview(Images.RGB, permutedims(convert.(N0f16, img./(2^12-1)), (3, 2,1))) 
println(size(img))
println(typeof(img))
println(size(img))
save("output_image.png", img)