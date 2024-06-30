import json

"""
Generates a json file with the required pinout
Used to keep some supplementary information

All pins BCM scheme

For x,y,z corresponds to the pins driving the x y and z motor in ms order
Motor_dir refers to rotation direction for positive moves, ie iterating step list forward or backward
Endstops contains for each motor one list with 2 pins corresponding to min max
Both Motor_dir and Endstops are in order x,y,z

Iteration when imaging goes as: x,y,z 

Image processing interprets y as left to right in image
And z as right to left
TODO: Above statement might be wrong -> but easy enough to change interface

Accelerometer: Not yet implemented order is [SDA,SCL] for I2C
TODO: Implement accelerometer
"""


pinout = {
    'x': [9,11,5,6], #In J4 : 13,14,21,22
    'y': [17,27,22,10], # In J4: 0,2,3,12
    'z': [13,19,26,24], # In J4, 23,24,25,5
    'motor_dir': [1,1,1],
    'Endstops': [[25,23],[8,7],[16,12]], 
    # In J4: [[4,6],[10,11],[27,26]]
    "Accelerometer": [2,3] # In J4: 8,9 -> SDA1,SCL1
}

with open('./Pinout.json', 'w', encoding='utf-8') as f:
    json.dump(pinout, f, ensure_ascii=False, indent=4)

print("Dumped Pinout to Pinout.json in current directory")