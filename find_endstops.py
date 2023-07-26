from PyQT_control import Grid_Handler
from ULN2003Pi import ULN2003
import RPi.GPIO as GPIO
import time
import sys

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

res = input('Press Y to start finding endstops, press N to move each motor 100 steps back and forth in order xyz, press xy and or z to select which axes to find endstops for')

do_all = False
do_x = False
do_y = False
do_z = False
do_move = False

if res == 'Y':
    do_all = True
elif res.lower() == 'N':
    do_move = True
else:
    if 'x' in res:
        do_x = True
    if 'y' in res:
        do_y = True
    if 'z' in res:
        do_z = True




if do_all or do_x:
    found_endstop = [[False]]
    while not found_endstop[0][0]:
        # Move 
        found_endstop = grid.move_dist([-1000])
        if found_endstop[0][0] and found_endstop[0][1]=='min' :
            break
        elif found_endstop[0][0] and found_endstop[0][1]=='max':
            raise Exception('Coordinate grid wrong!')
    print('Found X min')

    # Find maximum
    while True:
        # Move 
        found_endstop = grid.move_dist([1000])
        if found_endstop[0][0] and found_endstop[0][1]=='max':
            break
        elif found_endstop[0][0] and found_endstop[0][1]=='min':
            raise Exception('Coordinate grid wrong!')
    print('Found X max')
if  do_all or do_y:
    # Y-Axis
    # Find minimum
    while True:
        # Move 
        found_endstop = grid.move_dist([0,-1000])
        if found_endstop[1][0] and found_endstop[1][1]=='min':
            break
        elif found_endstop[1][0] and found_endstop[1][1]=='max':
            raise Exception('Coordinate grid wrong!')
    print('Found Y min')

    # Find maximum
    while True:
        # Move 
        found_endstop = grid.move_dist([0,1000])
        if found_endstop[1][0] and found_endstop[1][1]=='max':
            break
        elif found_endstop[1][0] and found_endstop[1][1]=='min':
            raise Exception('Coordinate grid wrong!')
    print('Found Y max')

if do_all or do_z:
    # Z-Axis
    # Find minimum
    while True:
        # Move 
        found_endstop = grid.move_dist([0,0,-1000])
        if found_endstop[2][0] and found_endstop[2][1]=='min':
            break
        elif found_endstop[2][0] and found_endstop[2][1]=='max':
            raise Exception('Coordinate grid wrong!')
    print('Found Z min')

    # Find maximum
    while True:
        # Move 
        found_endstop = grid.move_dist([0,0,1000])
        if found_endstop[2][0] and found_endstop[0][1]=='max':
            break
        elif found_endstop[2][0] and found_endstop[2][1]=='min':
            raise Exception('Coordinate grid wrong!')
    print('Found Z max')


    print('Found all endstops')
    print('Gridbounds:')
    print(grid.gridbounds)
    print('Zeropoint')
    print(grid.zeropoint)

if do_move:
    print("moving x")
    grid.move_dist([100])
    print('moving back')
    grid.move_dist([-100])
    print("moving y")
    grid.move_dist([0,100])
    print('moving back')
    grid.move_dist([0,-100])
    print("moving z")
    grid.move_dist([0,0,100])
    print('moving back')
    grid.move_dist([0,0,-100])

print('Goodbye')