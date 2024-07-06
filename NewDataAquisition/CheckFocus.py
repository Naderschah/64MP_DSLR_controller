"""
Pre Imaging script

Prints grid bounds and size

Moves grid to x=y=0 and z to max
Launches Camera Window
Then on continue terminates camera window and moves to x=y=z=0
Launches Camera Window

Used to determine subject positioning and camera height

TODO: Order of grid move might be switched around
"""
from Controler_Classes import Grid_Handler, Camera_Handler
from ULN2003Pi import ULN2003
import time, json, sys, os
from pathlib import Path

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
cam = Camera_Handler(disable_tuning=False, disable_autoexposure=False)
res = input("Have the endstops been calibrated? [Y/N]")
if res.lower() != 'y':
    print("Please calibrate and come back")
    sys.exit(0)
print("Moving to x_max,0,0")
grid.move_to_coord([grid.gridbounds[0],0,0])

print("Please align subject")
print("Type 'capture' to take an image and display it")
print("Type exit to continue with the focus at 0,0,0")
while True:
    res = input()
    if res == 'capture':
        cam.start()
        cam.capture_image(str(Path.home())+'/cam_check_capture.png')
        cam.stop()
        os.system("feh {}".format(str(Path.home())+'/cam_check_capture.png'))
    elif res == 'exit':
        break
    else:
        print("Command not recognized")


cam.stop_preview()
print("Moving to 0,0,0")
grid.move_to_coord([0,0,0])

print("Please align subject")
print("Type 'capture' to take an image and display it")
print("Type exit to terminate script")
while True:
    res = input()
    if res == 'capture':
        cam.start()
        cam.capture_image(str(Path.home())+'/cam_check_capture.png')
        cam.stop()
        os.system("feh {}".format(str(Path.home())+'/cam_check_capture.png'))
    elif res == 'exit':
        break
    else:
        print("Command not recognized")

grid.disable_all(gpio_pins)
