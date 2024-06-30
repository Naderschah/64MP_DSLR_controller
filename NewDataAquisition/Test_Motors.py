from Controler_Classes import Grid_Handler
from ULN2003Pi import ULN2003
import json, time

with open('./Pinout.json', 'r') as f:
    gpio_pins = json.load(f)

grid = Grid_Handler(motor_x=ULN2003.ULN2003(gpio_pins['x']), 
                    motor_y=ULN2003.ULN2003(gpio_pins['y']), 
                    motor_z=ULN2003.ULN2003(gpio_pins['z']), 
                    motor_dir = gpio_pins['motor_dir'], 
                    endstops = gpio_pins['Endstops'],
                    ingore_gridfile=True)
print("Set up Grid Controller")

print("Movement commands are formated as: <Motor><Steps>, where <Motor>=x,y,z and steps is some number")
print("To move 100 steps in the x direction x100, to move 100 in the opposite x-100")
print("enter exit to terminate (turns off motors)")
while True:
    try:
        command = input()
        if command == 'exit':
            break
        axes = {'x':0,'y':1,'z':2}[command[0]]
        steps = int(command[1:]) 
        move = [0,0,0]
        move[axes] = steps
        print('Moving')
        grid.move_dist(steps)
    
    except Exception as e:
        print("Invalid Command")
        print("Error:")
        print(e)


grid.disable_all()
