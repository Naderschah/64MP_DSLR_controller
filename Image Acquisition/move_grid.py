"""
Utility function to move grid while preserving position
"""
from PyQT_control import Grid_Handler
from ULN2003Pi import ULN2003
import RPi.GPIO as GPIO
import time
import sys

def gen_grid_handler():
    motor_dir = [1,1,1]
    gpio_pins = {'x': [19,5,0,11],
                     'y':[9,10,22,27],
                     'z':[17,4,3,2], 
                     'IR':None,
                     # Endstops are connected to normally closed (ie signal travels if not clicked)!
                     'Endstops': [[20,21],[16,12],[7,8]],
                     }
    grid = Grid_Handler(motor_x=ULN2003.ULN2003(gpio_pins['x']), motor_y=ULN2003.ULN2003(gpio_pins['y']), motor_z = ULN2003.ULN2003(gpio_pins['z']), 
                        # if -1 invert motor direction
                        motor_dir = motor_dir, endstops = gpio_pins['Endstops'])
    return grid
