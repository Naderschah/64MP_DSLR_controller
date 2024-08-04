import h5py
import numpy as np
import cv2
path = "/run/media/felix/3677421f-5daf-4ea7-ba33-31b09d11edcf/Images/img_8/20400_0_8192_exp32000.hdf5"
"""
WHY DOES THIS SHIT NOT WORK


"""
with h5py.File(path, 'r') as file:
    # Access the dataset 'image'
    image_data = file['image']
    # Convert the dataset to a numpy array
    image_array = np.array(image_data)#[:,:-28]
    print(image_array.shape)
    # We now unpack all bits
bit_array = np.unpackbits(image_array, axis=1,bitorder='big')


expanded_bit_array = np.zeros((3040, 16 * (bit_array.shape[1]) // 12), dtype='bool')
print(bit_array[0,:12])
for i in range(bit_array.shape[1] // 12):
    # Source indices
    start_idx = i * 12
    end_idx = start_idx + 12
    
    # Destination indices
    dest_start_idx = i * 16 +4   # Shift destination start index to right by 4 to add zeros on the left
    dest_end_idx = dest_start_idx +12#+ 12
    
    # Copy the 12 bits into the correct position
    expanded_bit_array[:, dest_start_idx:dest_end_idx] = bit_array[:, start_idx:end_idx]
print(expanded_bit_array.shape)

img = np.packbits(expanded_bit_array,axis=1,bitorder='little').view('uint16')

img = (img/(2**12-1)*(2**8-1)).astype('uint8')

print(img[0,:])

demosaiced_image = cv2.cvtColor(img, cv2.COLOR_BayerBG2BGR)
cv2.imwrite('demosaiced_image.png', demosaiced_image)


print("testing")
print("UInt8 : 2")
test = np.array([2]).astype('uint8')
print("Order Big")
print(np.unpackbits(test,bitorder='big'))
print("Order Little")
print(np.unpackbits(test,bitorder='little'))
print("Increasing size to UInt16")
expanded_bit_array = np.zeros((16), dtype='bool')
expanded_bit_array[-8:] = np.unpackbits(test,bitorder='big')
print("Append big on the right")
print(expanded_bit_array)
print(np.packbits(test,bitorder='big'))
expanded_bit_array = np.zeros((16), dtype='bool')
expanded_bit_array[:8] = np.unpackbits(test,bitorder='little')
print("Append little on the left")
print(expanded_bit_array)
print(np.packbits(test,bitorder='little'))
