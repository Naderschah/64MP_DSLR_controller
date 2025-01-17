from Controler_Classes import init_grid
import RPi.GPIO as GPIO
import json

grid,gpio_pins = init_grid(ingore_gridfile=True)

print("Assure that none of the endstops are currently triggered")
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

# found_endstop is a variable of grid that is returned on each grid move
found_endstop = [[False]]
if do_all or do_x:
    while not found_endstop[0][0]:
        # Move 
        found_endstop = grid.move_dist([-1000,0,0],ignore_endstop=False)
        if found_endstop[0][0] and found_endstop[0][1]=='min' :
            break
        elif found_endstop[0][0] and found_endstop[0][1]=='max':
            raise Exception('Coordinate grid wrong!')
    print('Found X min')

    # Find maximum
    while True:
        # Move 
        found_endstop = grid.move_dist([1000,0,0],ignore_endstop=False)
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
        found_endstop = grid.move_dist([0,-1000,0],ignore_endstop=False)
        if found_endstop[1][0] and found_endstop[1][1]=='min':
            break
        elif found_endstop[1][0] and found_endstop[1][1]=='max':
            raise Exception('Coordinate grid wrong!')
    print('Found Y min')

    # Find maximum
    while True:
        # Move 
        found_endstop = grid.move_dist([0,1000,0],ignore_endstop=False)
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
        found_endstop = grid.move_dist([0,0,-1000],ignore_endstop=False)
        if found_endstop[2][0] and found_endstop[2][1]=='min':
            break
        elif found_endstop[2][0] and found_endstop[2][1]=='max':
            raise Exception('Coordinate grid wrong!')
    print('Found Z min')

    # Find maximum
    while True:
        # Move 
        found_endstop = grid.move_dist([0,0,1000],ignore_endstop=False)
        if found_endstop[2][0] and found_endstop[2][1]=='max':
            print("Found endstop for z max")
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