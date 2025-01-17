from PIL import Image
import numpy as np 
import h5py
import os

path = './CCM/1015/'
save_path = './CCM/1015/tmp'

"""
To inspect specific images
"""


def load_image(path):
    with h5py.File(path,'r') as f:
        img = f["image"][()]
    img = img.view('uint16')
    size = img.shape
    img_reshape = np.zeros((size[0]//2, size[1]//2, 3))
    img_reshape[:,:,0] = img[1::2,1::2]
    img_reshape[:,:,1] = (img[::2,1::2]+img[1::2,::2])//2
    img_reshape[:,:,2] = img[::2,::2]
    return (img_reshape/(2**12-1)*(2**8-1)).astype('uint8') # Turn float 0->1
        

for i in os.listdir(path):
    #if '_8192_8192_exp32000' in i:
    if i[-5:] == 'hdf5':
        img = load_image(os.path.join(path, i))
        img = Image.fromarray(img, mode='RGB')
        img.save(os.path.join(save_path, i[:-5]+'.png'))