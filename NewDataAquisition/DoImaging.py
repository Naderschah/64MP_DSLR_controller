"""
Does the imaging routine

As command line options the following need to be provided
 - exposure
 - ISO
 - Grid maxima redefinition (for smaller things)
 - Overlap override
 - Magnification override (or specification)
 - resolution

 Formating is 
 <ident>=<val>

 TODO Enable res changes in camera
 TODO Rework bounds finding, make it so that the input is the minimal area covered
        Also check that this is within original gridbounds
 TODO Add time estimate, 100Hz stepping * exp + overhead per im 



 mm_per_step conversion, the motors are:



"""
from Controler_Classes import Grid_Handler, Camera_Handler
from ULN2003Pi import ULN2003
import time, json, sys, os
import numpy as np
import psutil
from pathlib import Path


start = time.time()
# Initiate all the controller classs
with open('./Pinout.json', 'r') as f:
    gpio_pins = json.load(f)

grid = Grid_Handler(motor_x=ULN2003.ULN2003(gpio_pins['x']), 
                    motor_y=ULN2003.ULN2003(gpio_pins['y']), 
                    motor_z=ULN2003.ULN2003(gpio_pins['z']), 
                    motor_dir = gpio_pins['motor_dir'], 
                    endstops = gpio_pins['Endstops'],
                    ingore_gridfile=False)

# Keep all control structures enabled to allow easy grid alignment
cam = Camera_Handler(disable_tuning=True, disable_autoexposure=True)


## Command line parsing
cmd_line_opts = sys.argv[1:]

exposure = None
iso = None
overlap = 0.2
magnification = 2
res = [3040, 4056]
# Compute mm per step 
motor_deg_per_step = 1/64/64*360
stage_mm_per_deg = 0.5/360
mm_per_step=motor_deg_per_step*stage_mm_per_deg
print("mm per motor step {}".format(mm_per_step))
step_size_x = 200

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
        grid.gridbounds[0] = float(i.split('=')[1])//mm_per_step
    elif i.startswith("grid_y"):
        grid.gridbounds[1] = float(i.split('=')[1])//mm_per_step
    elif i.startswith("grid_z"):
        grid.gridbounds[2] = float(i.split('=')[1])//mm_per_step
    elif i.startswith("overlap"):
        overlap = float(i.split('=')[1])
    elif i.startswith("mag"):
        magnification = int(i.split('=')[1])
    elif i.startswith("res_x"):
        res[0] = int(i.split('=')[1])
    elif i.startswith("res_y"):
        res[1] = int(i.split('=')[1])
        

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
dirs = os.listdir('/media/micro/3677421f-5daf-4ea7-ba33-31b09d11edcf/Images')
dirname = 'img'
count = 0 
while dirname+'_'+str(count) in dirs:
    count += 1
dirname = dirname+'_'+str(count)
dir = '/media/micro/3677421f-5daf-4ea7-ba33-31b09d11edcf/Images/'+dirname
os.mkdir(dir)
os.chdir(dir)
print('Changed directory to {}'.format(dir))

# Make imaging array, first grab camera data  TODO Grab what possible from the camera
px_size = 1.55*1e-3 
im_y_len =  res[0]*px_size # mm width
im_z_len =  res[1]*px_size

effective_steps_per_mm_in_image = 1/(magnification*mm_per_step)
# Steps to move overlap distance
steps = [i*px_size * effective_steps_per_mm_in_image * (1-overlap) for i in res]
steps = [step_size_x, *steps]
# Quick memory check 
print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
# Generate array holding 
coord_arr = [np.arange(0,grid.gridbounds[i], steps[i]).append(grid.gridbounds[i]).astype(int) for i in range(3)]
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

# Start camera for imaging
cam.start()


for e in exposure:
    cam.set_exp(e)
    for i in coord_arr[2]:
        for j in coord_arr[1]:
            for k in coord_arr[0]:
                grid.move_to_coord([k,j,i])
                time.sleep(0.01) 
                cam.capture_image('{}'.format('_'.join([str(i) for i in grid.pos]))+'_exp{}.png'.format(e))
            coord_arr[0] = coord_arr[0][::-1]
        coord_arr[1] = coord_arr[1][::-1]
    coord_arr[2] = coord_arr[2][::-1]

t_diff = time.time()-start
h = t_diff // 3600
m = (t_diff-3600*h) // 60
s = t_diff -3600*h - m*60
print("Completed in {}:{}:{}".format(h,m,int(s)))
os.system('echo "False" > {}'.format(os.path.abspath(str(Path.home())+"/imaging.txt")))

grid.disable_all(gpio_pins)
cam.stop()