from PyQT_control import Grid_Handler
from ULN2003Pi import ULN2003

motor_dir = [1,1,1]
gpio_pins = {'x': [19,5,0,11],
                 'y':[9,10,22,27],
                 'z':[17,4,3,2], 
                 'IR':None,
                 # Endstops are connected to normally closed (ie signal travels if not clicked)!
                 'Endstops': [[20,21],[16,12],[8,7]],
                 }
grid = Grid_Handler(motor_x=ULN2003.ULN2003(gpio_pins['x']), motor_y=ULN2003.ULN2003(gpio_pins['y']), motor_z = ULN2003.ULN2003(gpio_pins['z']), 
                    # if -1 invert motor direction
                    motor_dir = motor_dir, endstops = gpio_pins['Endstops'])

res = input('Press Y to start finding endstops, press N to move each motor 100 steps back and forth in order xyz')

if res.lower() == 'y':
    # X-Axis
    # Find minimum
    while True:
        # Move 
        found_endstop = grid.move_dist([-1000])
        input('type now ', found_endstop)
        print('left move')
        print(found_endstop)
        if found_endstop[0][0] and found_endstop[0][0]=='min':
            break
    print('Found X min')

    # Find maximum
    while True:
        # Move 
        found_endstop = grid.move_dist([1000])
        if found_endstop[0][0] and found_endstop[0][0]=='max':
            break
    print('Found X max')

    # Y-Axis
    # Find minimum
    while True:
        # Move 
        found_endstop = grid.move_dist([0,-1000])
        if found_endstop[1][0] and found_endstop[1][0]=='min':
            break
    print('Found Y min')

    # Find maximum
    while True:
        # Move 
        found_endstop = grid.move_dist([0,1000])
        if found_endstop[1][0] and found_endstop[1][0]=='max':
            break
    print('Found Y max')


    # Z-Axis
    # Find minimum
    while True:
        # Move 
        found_endstop = grid.move_dist([0,0,-1000])
        if found_endstop[2][0] and found_endstop[2][0]=='min':
            break
    print('Found Z min')

    # Find maximum
    while True:
        # Move 
        found_endstop = grid.move_dist([0,0,1000])
        if found_endstop[2][0] and found_endstop[0][0]=='max':
            break
    print('Found Z max')


    print('Found all endstops')
    print('Gridbounds:')
    print(grid.gridbounds)
    print('Zeropoint')
    print(grid.zeropoint)

elif res.lower() == 'n':
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
else:
    print('Goodbye')