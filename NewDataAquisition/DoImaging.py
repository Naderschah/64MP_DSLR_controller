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
 - NS : Attempts to save all data on mounted ssh path -- Only worth it over Ether

 Formating is 
 <ident>=<val>

 TODO Wait for acceleration and thread time data to see if full stepping is worth using

 mm_per_step conversion, the motors are: conv_to_mm(1) defautl function for this now



"""
from Controler_Classes import init_grid, Camera_Handler, Accelerometer, conv_to_mm
import time, json, sys, os
import numpy as np
import psutil
from pathlib import Path
import copy
import cv2
import h5py

def compute_contrast(image, kernel_size=9):
    # Generate LoG kernel
    blur = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    # Change CV_32F to whatever the datatype of the passed or gray image is
    laplacian = cv2.Laplacian(blur, cv2.CV_32F, ksize=kernel_size)
    laplacian = np.abs(laplacian) # Take absolute value
    # Calculate max, min, and mean
    return laplacian.max(), laplacian.min(), laplacian.mean()


start = time.time()
# Initiate all the controller classs
grid,gpio_pins = init_grid()

## Command line parsing
cmd_line_opts = sys.argv[1:]

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
# TODO Use computed after the profiling is done
# The below is most likely overkill go with 150? also how is focus depth defined gpt refuses to give me a usefull answer
#step_size_x = int(12 / mm_per_step ) # This is ~100 steps so double as many images now
step_size_x = 100


imging_bounds = copy.deepcopy(grid.gridbounds)
overwrote_path = False

for i in cmd_line_opts:
    if i.startswith('exp'):
        arg = i.split('=')[1]
        # Check multiple exposure
        if arg.startswith('['):
            exposure = [int(i) for i in arg[1:-2].split(',')]
        else: 
            exposure = [int(arg)]
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
    elif i.startswith("path"):
        img_path = i.split('=')[1]
        overwrote_path = True

# Check if drive is mounted
def is_drive_mounted(drive_path):
    partitions = psutil.disk_partitions()
    for partition in partitions:
        if partition.mountpoint == drive_path:
            return True
    return False

if not overwrote_path:
    if not is_drive_mounted(img_path):
        print("Please mount the remote drive, or specify a different path")
        sys.exit(0)

"""
Ok so some ground up weed for imaging, accept the misalignment for now
Tmr do black level, and nvm having a control, just enable ccm and see what happens, also disable cac
Make the image really small 


Testing order ::: Find Small Subject!
First make another image to check how for threading and precomputed contrast works, decide on final testing setup
Then activate black level < -- Shouldnt make much of a difference
Then disable AWB
Then CCM default config --> This one is prob pretty usefull
Then any others
Then acceleration based image rejection
Disable cac (Chromatic abberation correction) -> It just shifts green and blue 2 px

Disable Sharpen?

Try scientific version available -> Find source to see whats scientific about this one
- Modified ct_curve for awb
- Disabled rpi.contrast through ce_enable
- CCM actually measured -> More datapoints
- No mention of alsc

AGC (automatic gain/exposure correction) is not mentioned below > was it on the whoel time? doubt it Prob done through controls
AWB (auto white balance) not mentioned here or in custom controls
TODO: AWB was on all along, in custom controls i need to set "ColourGains" to (1.,1.) I guess it just says set to disable, but you can set between 0 and 32
sdn : Spatial denoise

Maybe just use raw stream and do processing after the fact?


New Order of Operation: python3 DoImaging.py exp=32000 iso=1 grid_x=5 grid_y=1 grid_z=1 # 206, 2, 2 images -> 4.5h
- Small image of ground up weed using the same config as always 1x1 mm in y and z, and 5 mm height to look at rejection 
- Next scientific image combining including the x coordinates included after rejection originally. 


It seems that the flex of the glue I added moves the motors during imaging, will investigate further

"""

#TODO Temporarily set tuning to true and manually disabling here to mess with parameters and see effects
#cam.tuning['algorithms'][0]['rpi.black_level']['black_level'] = 0 # Black level can probably be safely enabled -> Hopefully gets rid of some of that red background, applied to all can be overwritten with _r _g and _b 
#cam.tuning['algorithms'][4]['rpi.geq']['offset'] = 0 # Green equalisation -> Default valeus selected i think 
#cam.tuning['algorithms'][4]['rpi.geq']['slope'] = 0
## luminance 0 disables algorithms effect
#cam.tuning['algorithms'][8]['rpi.alsc']["luminance_strength"] = 0 # Auto lens shading correction : Can be calibrated, doesnt seem necessary
## Reduce load on isp
#cam.tuning['algorithms'][8]['rpi.alsc']["n_iter"] = 1 # Automatic Lens Shading Correction --> Not relevant for me
## Disable gamma curve
#cam.tuning['algorithms'][9]['rpi.contrast']["ce_enable"] = 0 # contrast and Gamma control? Maybe usefull but probably not, havent actually applied it after the fact never seems needed, maybe up ahead helps?
## Tuning file sometimes has sharpen and ccm swapped 
#if 'rpi.ccm' in cam.tuning['algorithms'][10]: index = 10  # Color Correction matrix see how well default works, see if can find calibration sources
#elif 'rpi.ccm' in cam.tuning['algorithms'][11]: index = 11
#
## Disable color correction matrix for all color temperatures 
#for i in range(len(cam.tuning['algorithms'][index]['rpi.ccm']['ccms'])):
#    cam.tuning['algorithms'][index]['rpi.ccm']['ccms'][i]['ccm'] = [1,0,0,0,1,0,0,0,1]
# END temp change
from picamera2 import Picamera2
# Load scientific tuning file
#tuning = Picamera2.load_tuning_file("/usr/share/libcamera/ipa/rpi/vc4/imx477_scientific.json")

# Keep all control structures enabled to allow easy grid alignment
cam = Camera_Handler(disable_tuning=False, 
                     disable_autoexposure=True, 
                     res={"size":(res[0],res[1])},)
                     #tuning_overwrite=tuning)
cam.stream = 'raw'

acc = Accelerometer()



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

# Mark imaging for rsync 
os.system('echo "True" > {}'.format(os.path.abspath(str(Path.home())+"/imaging.txt")))
# Make the imaging directory and move there
dirs = os.listdir(img_path)
dirname = 'img'
count = 0 
while dirname+'_'+str(count) in dirs:
    count += 1
dirname = dirname+'_'+str(count)
dir = os.path.join(img_path,dirname)
os.mkdir(dir)
os.chdir(dir)
print('Changed directory to {}'.format(dir))

# Make imaging array, first grab camera data
px_size = 1.55*1e-3 
im_y_len =  sensor_size[0]/magnification # mm width
im_z_len =  sensor_size[1]/magnification

effective_steps_per_mm_in_image = 1/(magnification*mm_per_step)
# Steps to move overlap distance
steps = [i* effective_steps_per_mm_in_image * (1-overlap) for i in sensor_size]
steps = [step_size_x, *steps]
# Quick memory check 
print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
# Generate array holding 
coord_arr = [np.append(np.arange(0,imging_bounds[i], steps[i]), imging_bounds[i]).astype(int) for i in range(3)]
# And again 
print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
# Quick print for imager
print("Image stop length per axis:")
for i in coord_arr:
    print(str(len(i))+": {}".format(i))

print("Imge stop length per axis in mm:")
for i in coord_arr:
    print(str(len(i))+": {}".format([j*mm_per_step for j in i]))
print("Camera image width and height in mm: {}, {}".format(im_y_len, im_z_len))


# Write grid to file for realtime processing
with h5py.File(os.path.join(dir, 'grid.hdf5'), "w") as f:
    xdat = f.create_dataset("x", data=coord_arr[0])
    ydat = f.create_dataset("y", data=coord_arr[1])
    zdat = f.create_dataset("z", data=coord_arr[2])
    expdat = f.create_dataset("exp", data=exposure)

# Start camera for imaging
cam.start()


# Time estimate

# Steps per second: 100Hz -> Call it 90 with all the sleep and checking
# Imaging time is e
# Overhead is?

max_dist = [np.max(i) for i in coord_arr]
move_multiplier = [1, len(coord_arr[2]),len(coord_arr[2])*len(coord_arr[1])]
tot_move_dist_for_axis = [max_dist[i]*move_multiplier[i] for i in range(len(max_dist))]
time_estimate_steps = [tot_move_dist_for_axis[i]/90 for i in range(len(exposure))]
tot_time = sum(time_estimate_steps) + sum([len(coord_arr[2])*len(coord_arr[1])*e for e in exposure])*1e-6


# Acc limits mean readout on still (abs) +- 1.5std (not abs) -> yzx since readout axes different
# measured during no move but motors on 10000 datapoints
deviations = 1.5
yzx_mean_readout = [ 1.23893293e-01,  8.19051408e-03, 9.21124513e+00]
yzx_std_readout = [0.24559896, 0.27919091, 0.34976954]


print("Assumed total time is {}".format(tot_time))
# TODO: Max accel filter for images
start = time.time()
with open(dir+'/meta.txt', 'a') as f:
    f.write("x,y,z,accel,time since start, contrast max, contrast min, contrast mean\n")
    for e in exposure:
        print("Exposure: {}".format(e))
        cam.set_exp(e)
        for i in coord_arr[2]:
            acc_val =  acc.get()
            while (np.abs(acc_val[0]) > yzx_mean_readout[0] + yzx_std_readout[0]) and (np.abs(acc_val[0]) < yzx_mean_readout[0] - yzx_std_readout[0]):
                #  Check at 100Hz until within bounds
                time.sleep(0.01)
            for j in coord_arr[1]:
                acc_val =  acc.get()
                while (np.abs(acc_val[1]) > yzx_mean_readout[1] + yzx_std_readout[1]) and (np.abs(acc_val[1]) < yzx_mean_readout[1] - yzx_std_readout[1]):
                    #  Check at 100Hz until within bounds
                    time.sleep(0.01)
                for k in coord_arr[0]: # Inner loop 2.4 - 2.6s
                    _start = time.time()
                    grid.move_to_coord([k,j,i]) # 0.5s for 100 steps
                    print([k,j,i])
                    acc_val =  acc.get()
                    while (np.abs(acc_val[2]) > yzx_mean_readout[2] + yzx_std_readout[2]) and (np.abs(acc_val[2]) < yzx_mean_readout[2] - yzx_std_readout[2]):
                        #  Check at 100Hz until within bounds
                        time.sleep(0.01)
                    img = cam.capture_array()# Return image for contrast profiling
                    # Wait till previous image saved, since switch to hdf5 should be much quicker
                    cam.wait_for_thread() # Thread takes 0.1 - 0.2 s
                    # start seperate thread to save image 
                    # --> Note deepcopy to avoid the new img overwriting the old
                    cam.threaded_save('{}'.format('_'.join([str(i) for i in grid.pos]))+'_exp{}.hdf5'.format(e), copy.deepcopy(img))
                    # Doing this instead of threading adds ~12 min for 12000 images, io more important 
                    # Copy so it is C-contiguous apparently it isnt at the moment
                    img = img[:, :-16].copy().view('uint16')
                    # Equal grey project with green averaged first
                    contrast_img = (img[::2,::2]//3 + img[1::2,::2]//6 + img[::2,1::2]//6+ img[1::2,1::2]//3).astype('uint16')
                    res = compute_contrast(contrast_img, kernel_size=9) 
                    f.write("{},{},{},{},{},{},{}\n".format(k,j,i,time.time()-start, res[0],res[1],res[2]))
                    f.flush() #Just in case
                    print("Time for loop {}".format(time.time()-_start))

                    """
                    Timing: 1.1 for inner loop (~0 if 1e-3)
                    move_coord : 0.5s (100steps)
                    accel.get : ? ~0?
                    image : 0.2s @ 32000 mu s exp -> 0.16s overhead
                    threaded_save : ~0
                    comupute_contrast : 0.32
                    write file : ~0
                    ----
                    That works out correctly

                    Acc never triggered, but kept to determine bumping the microscope

                    At 100 steps for full range it should be 14.685 minutes per y,z pos

                    Timing for network save over ether seems the same as normal
                    """

                coord_arr[0] = coord_arr[0][::-1]
            coord_arr[1] = coord_arr[1][::-1]
        coord_arr[2] = coord_arr[2][::-1]
t_diff = time.time()-start
h = t_diff // 3600
m = (t_diff-3600*h) // 60
s = (t_diff -3600*h - m*60) //1
print("Completed in {:02}:{:02}:{:02}".format(int(h),int(m),int(s)))
os.system('echo "False" > {}'.format(os.path.abspath(str(Path.home())+"/imaging.txt")))

# Add move away from min such that one may safely find endstops on the next run
print("Moving close to min points")
grid.move_to_coord([i//8 for i in grid.gridbounds])


grid.disable_all(gpio_pins)
cam.stop() 