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

 TODO Rework bounds finding, make it so that the input is the minimal area covered
        Also check that this is within original gridbounds
        What does this mean
 TODO Add time estimate, 100Hz stepping * exp + overhead per im 
        I think i did this and it didnt work, so I guess measure and estimate based on resolution?

 TODO So turns out the package utilizes half step by default, full step is implemented and supported
        So switch to that (check the packages source code)

 mm_per_step conversion, the motors are:



"""
from Controler_Classes import init_grid, Camera_Handler, Accelerometer, conv_to_mm
import time, json, sys, os
import numpy as np
import psutil
from pathlib import Path


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

step_size_x = 200
mm_per_step = conv_to_mm(1)

imging_bounds = grid.gridbounds

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


# Keep all control structures enabled to allow easy grid alignment
cam = Camera_Handler(disable_tuning=True, 
                     disable_autoexposure=True, 
                     res={"size":(res[0],res[1])})

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

# Make imaging array, first grab camera data
px_size = 1.55*1e-3 
im_y_len =  sensor_size[0]/magnification # mm width
im_z_len =  sensor_size[1]/magnification
#TODO Use sensor size instead

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

print("Assumed total time is {}".format(tot_time))

with open(dir+'/timing.txt', 'w') as t:
    with open(dir+'/meta.txt', 'w') as f:
        for e in exposure:
            print("Exposure: {}".format(e))
            cam.set_exp(e)
            for i in coord_arr[2]:
                for j in coord_arr[1]:
                    start = time.time()
                    for k in coord_arr[0]:
                        grid.move_to_coord([k,j,i])
                        time.sleep(0.01) 
                        f.write("{},{},{}:{}".format(k,j,i,sum(acc.get())))
                        cam.capture_image('{}'.format('_'.join([str(i) for i in grid.pos]))+'_exp{}.png'.format(e))
                        print([k,j,i])
                        f.write("{},{},{}:{}".format(k,j,i,sum(acc.get())))
                    coord_arr[0] = coord_arr[0][::-1]
                    t.write("{}".format(time.time()-start))
                coord_arr[1] = coord_arr[1][::-1]
            coord_arr[2] = coord_arr[2][::-1]

t_diff = time.time()-start
h = t_diff // 3600
m = (t_diff-3600*h) // 60
s = (t_diff -3600*h - m*60) //1
print("Completed in {}:{}:{}".format(h,m,int(s)))
os.system('echo "False" > {}'.format(os.path.abspath(str(Path.home())+"/imaging.txt")))

# Add move away from min such that one may safely find endstops on the next run
print("Moving close to min points")
grid.move_to_coord([i//8 for i in grid.gridbounds])


grid.disable_all(gpio_pins)
cam.stop() 