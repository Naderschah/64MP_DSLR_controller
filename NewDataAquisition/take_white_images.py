"""
Script that takes a bunch of images (as fast as it can)
Place something white on the stage, this can be used to determine CCM and pixel corrections
It saves them to a folder for later processing
Adjust iso and exp as needed below
"""

import os, time, copy
from Controler_Classes import Camera_Handler

iso = 1
exp = 14000
count = 50

save_path = "/home/micro/flat_images"

if not os.path.isdir(save_path):
    os.mkdir(save_path)

res = [4056,3040]
cam = Camera_Handler(disable_tuning=False, 
                     disable_autoexposure=True, 
                     res={"size":(res[0],res[1])})
cam.stream = 'raw'

cam.set_iso(iso)
cam.set_exp(exp)


for i in count:
    img = cam.capture_array()
    cam.wait_for_thread()
    cam.threaded_save(os.path.join(save_path, time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())+'hdf5'), copy.deepcopy(img))

print("Done")