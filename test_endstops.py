import RPi.GPIO as GPIO
from PyQT_control import Grid_Handler
from ULN2003Pi import ULN2003

motor_dir = [-1,1,1]
gpio_pins = {'x': [19,5,0,11],
                 'y':[17,4,3,2], 
                 'z':[9,10,22,27],
                 'IR':None,
                 # Endstops are connected to normally closed (ie signal travels if not clicked)!
                 'Endstops': {'x_min':[21,20], 'x_max':[16,12], 'y_min':[1,7],'y_max':[8,25], 'z_min':[24,23],'z_max':[18,15]},
                 }
import time


grid = Grid_Handler(motor_x=ULN2003.ULN2003(gpio_pins['x']), motor_y=ULN2003.ULN2003(gpio_pins['y']), motor_z = ULN2003.ULN2003(gpio_pins['z']), 
                    # if -1 invert motor direction
                    motor_dir = motor_dir, endstops = gpio_pins['Endstops'])

print('Ready')

while True:
    if any(grid.endstop_bool):
        print(grid.endstop_bool)
        print(['x_min','x_max','y_min','y_max','z_min','z_max'])
    
