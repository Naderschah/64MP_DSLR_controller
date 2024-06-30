"""
Launches a graphical window with a grid control interface to determine exposure and camera settings





Print after every move current position in steps and mm 
Camera controls should be added to text based modification of settings as seen in Test_Motors.py



"""
from Controler_Classes import Grid_Handler, Camera_Handler,print_grid
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

cam = Camera_Handler(disable_tuning=True, disable_autoexposure=True)

cam.start_preview()

print("Movement commands are formated as: <Motor><Steps>, where <Motor>=x,y,z and steps is some number")
print("To move 100 steps in the x direction x100, to move 100 in the opposite x-100")
print("The same idea with the camera but e for exposure and i for iso or rather analogue gain")
print("enter exit to terminate (turns off motors and camera)")
while True:
    try:
        command = input()
        if command == 'exit':
            break
        if command.startswith('x','y','z'):
            axes = {'x':0,'y':1,'z':2}[command[0]]
            steps = int(command[1:]) 
            move = [0,0,0]
            move[axes] = steps
            print('Moving')
            grid.move_dist(steps)
            print_grid(grid)
        elif command.startswith('e','i'):
            to_run = {'e':cam.set_exp, 'i':cam.set_iso}[command[0]]
            val = int(command[1:])
            to_run(val)
        else:
            raise Exception("Command Not recognized")
    
    except Exception as e:
        print("Invalid Command")
        print("Error:")
        print(e)

grid.disable_all()
cam.stop_preview()
cam.stop()