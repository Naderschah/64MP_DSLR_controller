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
import time, json

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
cam = Camera_Handler(disable_tuning=False, disable_autoexposure=True)

print("Moving to 0,0,z_max")
grid.move_dist([0,0,grid.gridbounds[2]])

cam.start_preview()

input("Type something to continue")

cam.stop_preview()
print("Moving to 0,0,0")
grid.move_dist([0,0,grid.gridbounds[2]])

cam.start_preview()
input("Type something to stop")

cam.stop_preview()
grid.disable_all(gpio_pins)
