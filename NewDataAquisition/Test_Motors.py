from Controler_Classes import init_grid
import json, time
import sys

# Defaul val
ingore_gridfile = False
#grid ignore for deleted/corrupted/non-existant grid files
if len(sys.argv)>1:
    if sys.argv[1].strip() == 'gridignore':
        ingore_gridfile = True
    else:
        print("Invalid option")
else:
    ingore_gridfile = False
    
grid,gpio_pins = init_grid(ingore_gridfile = ingore_gridfile)

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
        grid.move_dist(move)
        print('Done')
    
    except Exception as e:
        print("Invalid Command")
        print("Error:")
        print(e)


grid.disable_all(gpio_pins)
