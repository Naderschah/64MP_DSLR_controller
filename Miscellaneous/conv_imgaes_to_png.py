from PIL import Image
import numpy as np 
import h5py
import os

path = '/run/media/felix/3677421f-5daf-4ea7-ba33-31b09d11edcf/Images/img_3'
save_path = '/run/media/felix/3677421f-5daf-4ea7-ba33-31b09d11edcf/Images/img_3/tmp_pngs'

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
    if '_8192_8192_exp32000' in i:
        img = load_image(os.path.join(path, i))
        img = Image.fromarray(img, mode='RGB')
        img.save(os.path.join(save_path, i[:-5]+'.png'))