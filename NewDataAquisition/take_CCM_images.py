"""
Script that takes a bunch of images (as fast as it can)
Place something white on the stage, this can be used to determine CCM and pixel corrections
It saves them to a folder for later processing
Adjust iso and exp as needed below
"""

import os, datetime, copy, time
from Controler_Classes import Camera_Handler
import numpy as np

iso = 1
exp = 14000
count = 50

save_path = "/home/micro/ccm_images"
colors = sorted (["3020","5015","6001","1023","5010","2004","7042","7046","9006","7016","9003","9017","4005","5018","6018","8003","8007","7036","9010","1015","3027","3022","8004","7035","2003","5017", "1021", "7047", "6016","6017",'1020',"1019","1018","1017","1016","4002","4003","4004","4006","4007","6002","6003","6019","6020","6021","8000","7015"])

# Filter because i messed up somewhere in the 5000-6000s
colors = [i for i in colors if int(i)>5000]

if not os.path.isdir(save_path):
    os.mkdir(save_path)

res = [4056,3040]
cam = Camera_Handler(disable_tuning=False, 
                     disable_autoexposure=True, 
                     res={"size":(res[0],res[1])})
cam.stream = 'raw'

cam.set_iso(iso)
cam.set_exp(exp)
cam.start()
time.sleep(1)

for i in range(len(colors)):
    input("Next color is RAL {}, press enter when ready".format(colors[i]))
    new_path = os.path.join(save_path, colors[i])
    if not os.path.isdir(new_path): os.mkdir(new_path)
    for j in range(5):
        # Check correct exposure
        exp_ = exp
        while True:
            img = cam.capture_array()
            # First we compute some exposure metrics
            mean_ =  np.mean(img.view('uint16'), axis= (0,1))
            if any((mean_ > 0.75 * (2**12-1)) and (mean_ < 0.75 * (2**12-1))):
                cam.wait_for_thread()
                cam.threaded_save(os.path.join(new_path, datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")+'.hdf5'), copy.deepcopy(img))
                break
            else:
                # And adjust exposure slowly 
                exp_  = int(exp_ * 0.8/np.max(mean_) * 0.9)
                cam.set_exp(exp_)
                print("adjusted exp to {}".format(exp_))
                time.sleep(1) 
    print("RAL {} completed".format(colors[i]))

cam.wait_for_thread()
print("Done")
cam.stop()