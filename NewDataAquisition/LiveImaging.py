"""
Does the imaging routine

As command line options the following need to be provided
 - exp=<> exposure
 - iso=<> ISO
 - grid_x=<> 
 - grid_y=<> 
 - grid_z=<> Grid maxima redefinition (for smaller things)
 - overlap=<> Overlap override
 - mag=<> Magnification override (or specification)
 - res_x=<>
 - res_y=<> resolution

 Formating is 
 <ident>=<val>

 mm_per_step conversion, the motors are: conv_to_mm(1) defautl function for this now

Difference of this one is that exposure is done on the inner loop and every image is taken twice
The purpose of this is to elimenate any extraneous vibration and allow for a larger exposure range 
to be sampled. 

A full dataset only taking one image amounts to 600GB of data, meaning that it is not feasable to
run this without processing at the same time

For every set of images, a file is created with the set of images for this substack
the algorithm running on the processing device reads only these files, processes the images in the substack
and deletes the source images and lockfile

For this a new directory structure is utilized, starting from the main folder:
Images
- curr_img.txt : Contains the directory in which live processing will be handled
- img_n : The directory curr_img.txt points to, same reasoning as in the other script
    - taken_imgs : Contains the taken images
    - first_process : Contains the processed images
    - substacks : contains to be focused substacks
        - $x_$y_$z_meta.txt : Contains information for this substack

Differences:
- Doesnt write to meta.txt as live processing doesnt facilitate for pre defined contrast filtering
- New directory structure
- Doesnt use accelerometer
- Doesnt mark for rsync (irrelevant now anyway)
- Does not proceed with starting a stack if there is less than 120 GB available (102GB required per set + 17 for the stacked images)


Processing:
- Searches substacks folder for images to process
- Processes them deletes source and saves them to first_process

TODO: This may become io locked due to simulatneous reading and writing from raspi and PC 

"""
from Controler_Classes import init_grid, Camera_Handler, conv_to_mm
import time, json, sys, os, subprocess
import numpy as np
import psutil
from pathlib import Path
import copy
import cv2
import h5py


start = time.time()
# Initiate all the controller classs
grid,gpio_pins = init_grid()

## Command line parsing
cmd_line_opts = sys.argv[1:]

drive_identifier = b'10.42.0.1' # IP addr on ether 
exposure = None
iso = None
overlap = 0.2
magnification = 2
res = [4056,3040]
sensor_size = [7.564,5.467] # Sensor size in mm not px count
mm_per_step = conv_to_mm(1) # ~0.122 mu m per step or 122 nm per step
img_path = '/media/micro/3677421f-5daf-4ea7-ba33-31b09d11edcf/Images'
img_path = '/home/micro/RemoteStorage/'
# Focus depth is 12 mu m 
step_size_x = 100


imging_bounds = copy.deepcopy(grid.gridbounds)
for i in cmd_line_opts:
    if i.startswith('exp'):
        arg = i.split('=')[1]
        # Check multiple exposure
        if arg.startswith('['):
            exposure = [int(i) for i in arg[1:-2].split(',')]
        else: 
            exposure = [int(arg)//2, int(arg), int(int(arg)*1.5)]
    elif i.startswith('iso'):
        iso = int(i.split('=')[1])
    elif i.startswith("grid_x"):
        imging_bounds[0] = float(i.split('=')[1])//mm_per_step
    elif i.startswith("grid_y"):
        imging_bounds[1] = float(i.split('=')[1])//mm_per_step
    elif i.startswith("grid_z"):
        imging_bounds[2] = float(i.split('=')[1])//mm_per_step
    elif i.startswith("overlap"):
        overlap = float(i.split('=')[1])
    elif i.startswith("mag"):
        magnification = int(i.split('=')[1])
    elif i.startswith("res_x"):
        res[0] = int(i.split('=')[1])
    elif i.startswith("res_y"):
        res[1] = int(i.split('=')[1])


def is_drive_mounted(identifier=b'10.42.0.1'):
    """
    Uses df -h to check for available space
    identitfier needs to be a byte string!
    """
    # Check identifier
    avail = [i for i in subprocess.check_output('df -m', shell=True).split(b'\n') if identifier in i]
    return (len(avail) == 1)


if not is_drive_mounted(identifier=drive_identifier):
    print("Please mount the remote drive")
    sys.exit(0)

# Keep all control structures enabled to allow easy grid alignment
cam = Camera_Handler(disable_tuning=False, 
                     disable_autoexposure=True, 
                     res={"size":(res[0],res[1])},)
                     #tuning_overwrite=tuning)
cam.stream = 'raw'

for i in range(len(imging_bounds)):
    if imging_bounds[i] > grid.gridbounds[i]:
        print("Warning: Imaging bounds larger than gridbounds!")
        print("Maximum Gridbounds:")
        print(grid.gridbounds)
        print("And in millimeters")
        print([i//mm_per_step for i in grid.gridbounds])
        sys.exit(0)

if exposure is None:
    print("Specify the exposure time in ms")
    sys.exit(0)
    
if iso is None:
    print("Specify the iso")
    sys.exit(0)
else:
    cam.set_iso(iso)

print("Initiated Camera and Motor Objects")

#           Proceed to generate imaging grid, metadata and start imaging

# Make the imaging directory and move there
dirs = os.listdir(img_path)
dirname = 'img'
count = 0 
while dirname+'_'+str(count) in dirs:
    count += 1
dirname = dirname+'_'+str(count)
dir = os.path.join(img_path,dirname)
# Write which directory is current
with open(os.path.join(img_path, 'curr_img.txt'), 'w') as f:
    f.write(dirname)

os.mkdir(dir)
os.chdir(dir)
print('Changed directory to {}'.format(dir))
"""
- curr_img.txt : Contains the directory in which live processing will be handled
- img_n : The directory curr_img.txt points to, same reasoning as in the other script
    - taken_imgs : Contains the taken images
    - first_process : Contains the processed images
    - substacks : contains to be focused substacks
        - $x_$y_$z_meta.txt : Contains information for this substack
"""
img_save_path = os.path.join(dir, 'taken_imgs')
os.mkdir(img_save_path)
processed_images = os.path.join(dir, 'first_process')
os.mkdir(processed_images)
substacks = os.path.join(dir, 'substacks')
os.mkdir(substacks)

# Make imaging array, first grab camera data
px_size = 1.55*1e-3 
im_y_len =  sensor_size[0]/magnification # mm width
im_z_len =  sensor_size[1]/magnification

effective_steps_per_mm_in_image = 1/(magnification*mm_per_step)
# Steps to move overlap distance
steps = [i* effective_steps_per_mm_in_image * (1-overlap) for i in sensor_size]
steps = [step_size_x, *steps]
# Generate array holding 
coord_arr = [np.append(np.arange(0,imging_bounds[i], steps[i]), imging_bounds[i]).astype(int) for i in range(3)]
# Quick print for imager
print("Image stop length per axis:")
for i in coord_arr:
    print(str(len(i))+": {}".format(i))

print("Imge stop length per axis in mm:")
for i in coord_arr:
    print(str(len(i))+": {}".format([j*mm_per_step for j in i]))
print("Camera image width and height in mm: {}, {}".format(im_y_len, im_z_len))

# Write grid to file for realtime processing -> Not used for that but still seems usefull to keep
with h5py.File(os.path.join(dir, 'grid.hdf5'), "w") as f:
    xdat = f.create_dataset("x", data=coord_arr[0])
    ydat = f.create_dataset("y", data=coord_arr[1])
    zdat = f.create_dataset("z", data=coord_arr[2])
    expdat = f.create_dataset("exp", data=exposure)

# Start camera for imaging
cam.start()

# FIXME: This is now wrong again
# Just compute based on ususal average time 1.2 normal add overhead for y and z moves
n_im = len(exposure) * 2 # last factor is img degeneracy
tot_time = 1.5*len(coord_arr[0])*len(coord_arr[1])*len(coord_arr[2]) * n_im
# For full set took 9:48 h and 650GB -> Single Image

##      Hard drive checking
def check_drive_space(identifier=b'10.42.0.1'):
    """
    Uses df -h to check for available space
    identitfier needs to be a byte string!
    """
    # <number> Megabytes
    avail = int([i for i in [i for i in subprocess.check_output('df -m', shell=True).split(b'\n') if identifier in i][0].split(b' ') if i != b''][-3].decode('utf-8'))
    return avail // 1024 # Convert to GB



print("Assumed total time is {}".format(tot_time))
start = time.time()
for i in coord_arr[2]:
    for j in coord_arr[1]:
        # Check ssd
        while check_drive_space(identifier= drive_identifier) <= 120:
            print("Drive Full, waiting")
            time.sleep(5) # Sleep for 5 seconds -> Processing may take quite a while
        __start = time.time()
        for k in coord_arr[0]: 
            grid.move_to_coord([k,j,i]) 
            print([k,j,i])
            imgs_for_substack = []
            for e in exposure: 
                cam.set_exp(e)
                # We now have to wait for the exposure to be applied to the next frame
                _start = time.time()
                framecnt = 0
                while True:
                    # The below returns the most recent frame metadata or 
                    # Waits for the next frame if previous has been returned
                    if int(cam.check_metadata()["ExposureTime"]) == int(e):
                        break # loop when settings set -> Expecting a 0.15s overhead one frame plus mistiming
                    framecnt += 1
                    print("Waited {} frames".format(framecnt)) #TODO : Temp debug
                    # Frame rate in used configuration: 10 fps : No need to wait the above is blocking
                print("Waited {} s for exp to be applied".format(time.time()-_start))
                # Take two images per exposure
                for ctr in (0, 1):
                    img = cam.capture_array()# Return image for contrast profiling
                    # Wait till previous image saved 
                    cam.wait_for_thread()  
                    # start seperate thread to save image --> Note deepcopy to avoid the new img overwriting the old
                    fname = '{}'.format('_'.join([str(ip) for ip in grid.pos]))+'_{}'.format(ctr)+'_exp{}.hdf5'.format(e)
                    cam.threaded_save(os.path.join(img_save_path,fname), copy.deepcopy(img))
                    imgs_for_substack.append(fname)
            # Invert exposure so that we dont have to wait for the next frame
            exposure = exposure[::-1]
            # Write stack file
            with open(os.path.join(substacks, '{}'.format('_'.join([str(i) for i in grid.pos]))+'_meta.txt') ,'w') as f:
                f.write("\n".join(imgs_for_substack))
            # And continue
        coord_arr[0] = coord_arr[0][::-1]
        print("Took {} s for x substack".format(time.time()-__start))
    coord_arr[1] = coord_arr[1][::-1]
coord_arr[2] = coord_arr[2][::-1]
t_diff = time.time()-start
h = t_diff // 3600
m = (t_diff-3600*h) // 60
s = (t_diff -3600*h - m*60) //1
print("Completed in {:02}:{:02}:{:02}".format(int(h),int(m),int(s)))
# Write to file to note that imaging is done
with open(os.path.join(dirname, 'curr_img.txt'), 'w') as f:
    f.write("")

# Add move away from min such that one may safely find endstops on the next run
print("Moving close to min points")
grid.move_to_coord([i//8 for i in grid.gridbounds])


grid.disable_all(gpio_pins)
cam.stop() 