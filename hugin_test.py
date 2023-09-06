#Preamble for restarts

import numpy as np
import subprocess
import os
import shutil
import imageio
import shutil
from calibrate import im_to_dat_with_gamma
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing
import sys
# First we create the proper file structure, we make acopy as this is just helper to the actual julia scripts
# record data size (GB) vs time taken (s)
time_taken = {75:0}
file_dict = {}
im_dir = "/media/felix/Drive/Images/img_5"
for i in os.listdir(im_dir): # 39000_60814_40569_NoIR.dng
    x,y,z, descriptor = i.split("_")
    descriptor = descriptor.split(".")
    # For all hdr exp the exposure is written as time.0mus.dng cause float
    if len(descriptor) == 2:
        descriptor = descriptor[0]
    else:
        descriptor = '.'.join(descriptor[0:2])
    if descriptor not in file_dict.keys():
        file_dict[descriptor] = {}
    if z not in file_dict[descriptor].keys():
        file_dict[descriptor][z] = {}
    if y  not in file_dict[descriptor][z].keys():
        file_dict[descriptor][z][y] = []
    file_dict[descriptor][z][y].append(x)


# record data size (only hdr folder here) (GB) vs time taken (s) --- actual focus stacking
time_taken = {65:0}
gray_projection = 'l-star'
contrast_window_size = 10
contrast_edge_scale = 0.3


save_path = os.path.join("/home/felix/hugin_processing", 'focused')
mask_dir = os.path.join("/home/felix/hugin_processing", 'masks')


if not os.path.isdir(save_path):
    os.mkdir(save_path)
if not os.path.isdir(mask_dir):
    os.mkdir(mask_dir)
# Change to this directory as masks are written to current directory
os.chdir(mask_dir)
# Again we only care about the subset
sub_keys = file_dict[list(file_dict.keys())[0]]
for z in sub_keys.keys():
    for y in sub_keys[z].keys():
        # Generate path
        #if HDR:
        #    in_path = os.path.join(use_path, 'HDR', z, y)+'/*'
        #else:
        #    in_path = os.path.join(use_path, file_dict.keys()[0], z, y)+'/*'
        in_path = os.path.join("/home/felix/hugin_processing", 'HDR', z, y)+'/*'
        # Generate command
        #cmd = 'enfuse -o {} --save-masks=%f_soft%E:{}_{}%f_mask%E --exposure-weight=0 --saturation-weight=0 --contrast-weight=1 --hard-mask --gray-projector={} --contrast-window-size={} --contrast-edge-scale={} {}'.format(out_path,z, y,gray_projection,contrast_window_size,contrast_edge_scale, in_path)
        #os.system(cmd)

        # Threaded testing of parameters
        def run_cmd(i):
            out_path = '{}/0_0_{}.tiff'.format(os.path.join("/home/felix/hugin_processing", 'focused'), i)
            in_path = os.path.join("/home/felix/hugin_processing", 'HDR', '0', '0')+'/*'
            gray_projection = 'l-star'
            contrast_edge_scale = -0.3
            cmd = 'enfuse -o {} --exposure-weight=0 --saturation-weight=0 --contrast-weight=1 --hard-mask --gray-projector={} --contrast-window-size={} --contrast-edge-scale={} --contrast-min-curvature={} {}'.format(out_path,gray_projection,7,-0.3,i/10*(2**16-1), in_path)
            os.system(cmd)
            return
        iterable = (1,2,3,4,5,6,7,8,9,10)
        pool = ThreadPool(multiprocessing.cpu_count()//4)
        data_arrays = pool.map(run_cmd, iterable)
        pool.close()
        pool.join()


        #for i in iterable:
        #    out_path = '{}/{}_{}_{}.tiff'.format(save_path,z, y, i)
        #    cmd = 'enfuse -o {} --exposure-weight=0 --saturation-weight=0 --contrast-weight=1 --hard-mask --gray-projector={} --contrast-window-size={} --contrast-edge-scale={} {}'.format(out_path,gray_projection,i,contrast_edge_scale, in_path)
        #    os.system(cmd)
        sys.exit(0)

"""
Testing so far,
contrast_edge_scale=0.3 --> Absolutely useless raises warning with:  "--contrast-min-curvature" is non-positive
contrast_edge_scale=-0.3 --> 7 window size is best but the issue arrises that with differing distances the parts of the subject arent in the same location need to find a way to deal with it --> align image stack doesnt work as there are no true control points
varying contrast-min-curvature --> 1-10 % does nothing
"""
