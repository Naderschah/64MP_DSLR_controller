import sys
import json
import time
try:
    from FinalDataProcessing.Controller_Classes.py import init_grid, Accelerometer
except:
    print("Make sure to move this file one dir up so that imports work")
    sys.exit(0)

increment = 1200 
start = 200
stop = 10000
current  = start
moves = []

while stop > current:
    moves.append(current)
    current += increment

# Make it some value that is smaller than all gridbounds
if sum(moves) > 50000:
    print("Total move is too large: {}".format(sum(moves)))
    sys.exit(0)

acc = Accelerometer()
grid,gpio_pins = init_grid()

# Move to zeropoint
grid.move_to_coord([0,0,0])
timing = {'x':None, 'y':None}
_timing = {i:None for i in moves}
__timing = []
x_coord = 0
for i in moves:
    __timing = []
    x_coord += i
    grid.move_to_coord([x_coord,0,0])
    start = time.time()
    while time.time()-start < 2: # Check for 2 seconds
        __timing.append([time.time()-start, acc.get()])
    _timing[i]=__timing

timing['x'] = _timing

# Now lets do y
y_coord = 0
for i in moves:
    __timing = []
    y_coord += i
    grid.move_to_coord([x_coord,y_coord,0])
    start = time.time()
    while time.time()-start < 2: # Check for 2 seconds
        __timing.append([time.time()-start, acc.get()])
    _timing[i]=__timing

timing['y'] = _timing


with open('TimingAndAcceleration.json', 'w') as f:
    json.dump(timing, f)
