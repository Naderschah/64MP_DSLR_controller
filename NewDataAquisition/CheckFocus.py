"""
Pre Imaging script

Prints grid bounds and size

Moves grid to x=y=0 and z to max
Launches Camera Window
Then on continue terminates camera window and moves to x=y=z=0
Launches Camera Window

Used to determine subject positioning and camera height
"""
from Controler_Classes import init_grid, Camera_Handler, conv_to_mm
import time, json, sys, os
from pathlib import Path

# Initiate all the controller classs
grid,gpio_pins = init_grid()

# Keep all control structures enabled to allow easy grid alignment
cam = Camera_Handler(disable_tuning=False, disable_autoexposure=False,low_res=False)
res = input("Have the endstops been calibrated? [Y/N]")
if res.lower() != 'y':
    print("Please calibrate and come back")
    sys.exit(0)
print("normal set up base in focus ~148.5 mm")
print("Moving to 0,0,0")
grid.move_to_coord([0,0,0])

print("Please align subject")
print("Type 'capture' to take an image and display it")
print("Type exit to continue with the focus at x_max,0,0")
print("Type exitnow to terminate now")
while True:
    res = input()
    if res == 'capture' or res == 'c':
        cam.start()
        cam.capture_image('/home/micro/RemoteStorage/cam_check_capture.png')
        cam.stop()
        print("Ready")
    elif res == 'exit':
        break
    elif res == 'exitnow':
        sys.exit(0)
    else:
        print("Command not recognized")


cam.stop_preview()
print("Moving to x_max,0,0")

grid.move_to_coord([grid.gridbounds[0],0,0])


print("Please align subject")
print("Type 'capture' to take an image and display it")
print("Type exit to terminate script")
while True:
    res = input()
    if res == 'capture' or res == 'c':
        cam.start()
        cam.capture_image(str(Path.home())+'/cam_check_capture.png')
        cam.stop()
        os.system("feh {}".format(str(Path.home())+'/cam_check_capture.png'))
    elif res == 'exit':
        break
    else:
        print("Command not recognized")

grid.disable_all(gpio_pins)
# Print gridsize for subject alignment
motor_deg_per_step = 1/64/64*360
stage_mm_per_deg = 0.5/360
mm_per_step=motor_deg_per_step*stage_mm_per_deg
print("Grid size in millimeters is approximately ", [i*mm_per_step for i in grid.gridbounds])