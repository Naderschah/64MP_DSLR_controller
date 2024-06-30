import os
import json
import sys
import time
import RPi.GPIO as GPIO
from picamera2 import Picamera2, Preview
from libcamera import controls
from tabulate import tabulate

class Grid_Handler:
    """
    Class to keep track of the imaging grid

    dir: 0 --> Towards camera

    """
    # The displacement is measured in ms so the grid is step size dependent
    # Keep track of bounds
    n_motors = 3
    gridbounds = [0]*n_motors
    zeropoint = [0]*n_motors
    # Position in grid
    pos = [0]*n_motors
    tot_move = [0]*n_motors
    last_pos = [0]*n_motors
    dx = 1
    pos = None
    # Bool to check if zeropoint set
    zero_made = False
    def __init__(self,motor_x,motor_y,motor_z,motor_dir,dx, endstops,ingore_gridfile) -> None:
        
        self.dx = dx
        self.motor_dir= motor_dir
        # Motors
        self.x = motor_x
        self.y = motor_y
        self.z = motor_z
        self.motors = [self.x, self.y, self.z]

        # Load gridbound and pos (with backup because its annoying to calibrate it again)
        if os.path.isfile(os.path.join(os.environ['HOME'], 'grid')):
            try:
                with open(os.path.join(os.environ['HOME'], 'grid'), 'r') as f:
                    cont = json.loads(f.read())
            except:
                print('Grid file corrupted attempting grid backup file')
                with open(os.path.join(os.environ['HOME'], 'grid_backup'), 'r') as f:
                    cont = json.loads(f.read())
            
            # Load and assign variables
            self.zeropoint = self.load_from_json('zeropoint',cont)
            self.gridbounds = self.load_from_json('gridbounds',cont)
            self.pos = self.load_from_json('pos',cont)
        elif not ingore_gridfile:
            print('No grid file found, recalibration required')
            sys.exit(0)
        else:
            print("No gridfile initiated watch out that nothing breaks")
            self.gridbounds = [0,0,0]
            self.zeropoint = [0,0,0]

        # Set up endstops
        self.endstops = endstops
        # Set up endstops
        GPIO.setmode(GPIO.BCM)
        for key in endstops:
            GPIO.setup(key[0], GPIO.IN, pull_up_down=GPIO.PUD_UP)
            GPIO.setup(key[1], GPIO.IN, pull_up_down=GPIO.PUD_UP)
            time.sleep(0.01)

        return
    

    def load_from_json(self, name, json_):
        var_ = None
        if 'gridbounds' in json_: 
            var_ = json_[name]
        else: 
            print("No {} found in file, recalibration required, exiting".format(name))
            sys.exit(0)
        return var_
        
    def set_gridbounds(self,bounds):
        self.gridbounds = bounds
    
    def set_pos(self,pos):
        self.pos = pos

    def reset_grid(self,axis=None):
        """Resets the grid
        
        TODO: Not quite sure if this is needed
        """
        if axis == None:
            self.zero_made = [False]*3
        else:
            self.zero_made[axis] = False
        return

    def make_zeropoint(self,axis): 
        """
        Set zeropoint for axis
        """
        self.zeropoint[axis] = 0
        self.gridbounds[axis] += self.pos[axis]
        self.pos[axis] = 0
        self.zero_made[axis] = True
        return

    def make_endstop(self, axis):
        """
        axis : int describing axes x:0, y:1, z:2 
        """
        self.gridbounds[axis] = self.pos[axis]

        return 

    def move_dist(self,disp,check_interval = 200,ignore_endstop=True):
        """
        disp --> displacement in motor steps list of 3 elements, one for each motor
        check_interval corresponds to movement intervals between which endstops are checked
        ignore_endstop is used for finding endstops, during imaging often the endstop may be
        triggered just before the defined location is found, to avoid inconsistencies in the imaging grid
        the detected endstops are by default ignored, however, this can not be done during finding the endstops
        """
        # Adjust sign
        for i in range(3):
            disp[i] = disp[i]*self.motor_dir[i]
        # Save last state 
        self.last_pos = self.pos
        # Do movement
        found_endstop = [ [False, False], [False, False], [False, False]]
        
        for i in range(len(disp)):
            if disp[i] != 0 or disp[i] != 0.0:
                # We split the movement into multiples of check_interval
                moved = 0
                sign = int(disp[i]/abs(disp[i]))
                while abs(moved) < abs(disp[i])-check_interval:
                    # Check the endstops and move accordingly
                    if (GPIO.input(self.endstops[i][0]) and disp[i]<0) and not ignore_endstop:
                        self.make_zeropoint(axis=i) 
                        print('Made zeropoint based on endstop in axis {}'.format(i))
                        found_endstop[i] = [True, 'min']
                        # Break while loop overwrite disp and continue
                        disp[i] = moved
                        print('Updated disp to ', disp)
                        break
                    elif (GPIO.input(self.endstops[i][1]) and disp[i]>0) and not ignore_endstop:
                        self.make_endstop(axis=i)
                        print('Made max point based on endstop')
                        found_endstop[i] = [True, 'max']
                        # Break while loop overwrite disp and continue
                        disp[i] = moved
                        break
                    
                    self.motors[i].step(sign*check_interval*self.motor_dir[i])
                    moved += sign*check_interval*self.motor_dir[i]
                # Once the above terminates we still need to move the remainder
                if disp[i] - moved != 0 and not found_endstop[i][0]:
                    self.motors[i].step(disp[i] - moved)
        # For book keeping undo motor correction
        for i in range(3):
            disp[i] = disp[i]*self.motor_dir[i]
        # Save new pos and tot move
        for i in range(len(disp)):
            self.pos[i] = self.pos[i]+disp[i]
            self.tot_move[i] = self.tot_move[i]+disp[i]
        # Save new coordinate -> In two files, so that if one is broken the other can be used -> travel error will depend on step size
        with open(os.path.join(os.environ['HOME'], 'grid_backup'),'w') as f:
            f.write(json.dumps({'pos':self.pos, 'gridbounds':self.gridbounds, 'zeropoint': self.zeropoint}))
        with open(os.path.join(os.environ['HOME'], 'grid'),'w') as f:
            f.write(json.dumps({'pos':self.pos, 'gridbounds':self.gridbounds, 'zeropoint': self.zeropoint}))
        if any([i[0] for i in found_endstop]) : print('Completed finding endstop in move')

        return found_endstop

    def move_to_coord(self,coord):
        """
        coord: [x,...] list with coords to go to
        """
        # Get coord difference
        disp = [coord[i]-self.pos[i] for i in range(len(coord))]
        self.move_dist(disp, adjust_ms=False)
        return

    def disable_all(self,gpio_pins):
        # Disable pins
        pins = []
        if gpio_pins['x'] is not None:
            pins += gpio_pins['x']
        if gpio_pins['y'] is not None:
            pins += gpio_pins['y']
        if gpio_pins['z'] is not None:
            pins += gpio_pins['z']
        for i in pins: GPIO.output(i,0)
        return
        
class Camera_Handler:
    custom_controls = { "AeEnable": False,  # Auto Exposure Value
                        "AwbEnable":False,  # Auto White Balance
                        "ExposureValue":0, # No exposure Val compensation --> Shouldnt be required as AeEnable:False
                        "NoiseReductionMode": controls.draft.NoiseReductionModeEnum.Off}
    camera = None # Set by configure_and_start
    iso = None
    exp = None
    still_config = None


    def __init__(self, disable_tuning=True, disable_autoexposure=True):
        # We start by disabling most of the algorithms done to change the image
        # Also initiates the self.camera object
        self.disable_algos(disable_tuning)

        self.configure_and_start(disable_autoexposure)
        self.disable_autoexposure = disable_autoexposure
        return

    def disable_algos(self,disable_tuning):
        """Disable everything that may interfere with the RAW images"""
        # Load tuning file
        self.tuning = Picamera2.load_tuning_file("/usr/share/libcamera/ipa/rpi/vc4/imx477.json")
        if disable_tuning:
            self.tuning['algorithms'][0]['rpi.black_level']['black_level'] = 0
            self.tuning['algorithms'][4]['rpi.geq']['offset'] = 0
            self.tuning['algorithms'][4]['rpi.geq']['slope'] = 0
            # luminance 0 disables algorithms effect
            self.tuning['algorithms'][8]['rpi.alsc']["luminance_strength"] = 0
            # Reduce load on isp
            self.tuning['algorithms'][8]['rpi.alsc']["n_iter"] = 1
            # Disable gamma curve
            self.tuning['algorithms'][9]['rpi.contrast']["ce_enable"] = 0
            # Tuning file sometimes has sharpen and ccm swapped 
            if 'rpi.ccm' in self.tuning['algorithms'][10]: index = 10 #TODO : Make all independent of index
            elif 'rpi.ccm' in self.tuning['algorithms'][11]: index = 11

            # Disable color correction matrix for all color temperatures
            for i in range(len(self.tuning['algorithms'][index]['rpi.ccm']['ccms'])):
                self.tuning['algorithms'][index]['rpi.ccm']['ccms'][i]['ccm'] = [1,0,0,0,1,0,0,0,1]
        
        return

    def configure_and_start(self,disable_autoexposure):
        # Start camera with tuning file
        self.camera = Picamera2(tuning=self.tuning)
        # Retrieve relevant configuration options
        self.still_config = self.camera.create_still_configuration(raw={})
        # Overwrite with custom control attriubtes
        if disable_autoexposure:
            for key in self.custom_controls:
                self.still_config[key] = self.custom_controls[key] 
        #And set custom controls (stop and start so that the next frame indeed has the correct controls)
        self.camera.stop()
        self.camera.set_controls(self.custom_controls)
        self.camera.start()

    def set_iso(self,iso):
        self.camera.set_controls({'AnalogueGain':iso})
        self.iso = iso
        return iso

    def set_exp(self,exp):
        self.camera.set_controls({'ExposureTime':exp})
        self.exp = exp
        return exp
    
    def capture_image(self,path):
        self.camera.capture_file(path)
        return
    
    def start_preview(self,res=(720,480)):
        self.camera.configure(self.camera.create_preview_configuration(queue=False ,main={"size":res})) 
        if self.disable_autoexposure:
            self.camera.set_controls(self.custom_controls)
        self.camera.stop()
        self.camera.start_preview(Preview.QT) # Should be non blocking
        self.camera.start()

    def stop_preview(self,):
        self.camera.stop_preview()
        self.camera.stop()
        self.camera.start()
    
    def stop(self):
        self.camera.stop()

    

def print_grid(grid,mm_per_step=0.00012397): # TODO : Number may be wrong
    """Takes the grid controller as an input and prints the current position 
    and grid bounds in both steps an millimeters"""
    table = [["Pos steps",*grid.pos], ["Pos mm", *[i*mm_per_step for i in grid.pos]], 
             ["Gb steps", *grid.gridbounds], ["Gb mm",*[i*mm_per_step for i in grid.gridbounds]]]
    
    print(tabulate(table))
    return


