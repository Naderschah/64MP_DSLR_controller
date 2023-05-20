
import typing
from PyQt5.QtWidgets import QWidget
from UI_Viewfinder import Ui_Viewfinder
from UI_config_window import Ui_MainWindow
from UI_endstop_window import Ui_Endstop_window
from PyQt5 import QtWidgets, QtCore, QtGui
import sys, subprocess
from picamera2 import Picamera2
from picamera2.previews.qt import QGlPicamera2
import numpy as np
import time 
import os
import datetime as dt
import logging
from pathlib import Path
from libcamera import controls
import threading
import RPi.GPIO as GPIO
from functools import partial
# Controller imports
from inputs import get_gamepad
import math
ZoomLevels = (1)


"""

TODOs: 
- Make sidebar QBoxLayout rather than widget
- SideBar background inconsisten color --> Make transparent
- centralWidget background --> make black
- Make buttons larger
- Add white balance sliders
- Figure out average imaging time 


When opening endstop first and hten graphical Gain choices wrong (fractions) -> find out why

Change comments in motor control about direction, current implementation correct, somewhere in reasoning i messed up

X: step res
1 step @ MS1 == 0.004 mm
Pred:
1 step @ MS16 ==  0.00025 mm 
Measured:
1 step @ MS16 ==  ? mm 


Vertical and Horizontal res
 1.55 μm × 1.55 μm pixel size
 7.9 mm diag, horiz: 6.2868 mm x 4.712 mm

with 2x:
 each px sees 0.775x0.775 mu m
 the sensor sees 3.1434 mm x 2.356

with 4x:
 each px sees 0.3875x0.3875 mu m
 the sensor sees 1.5717 mm x 1.178

"""



# Setting CMA to 1024 doesnt work 512 does apparently CMA is required to come from the bottom 1GB of ram, which is shared with kernel gpu 


# Note all setting changes after camera start take a little to be enables (1-5s)
# For 64MP all sensor modes have a different crop limit

# Docs on NoiseReductionMode wrong (min:0 max:4 -> int) on site 3 options -> str : Assume 0 is no noise reduction
# WHat does Noise Reduction do?
logging.basicConfig(filename=str(Path.home())+'/Camera/logs/{}.log'.format(dt.datetime.now().strftime('%Y%m%d')), filemode='w', level=logging.DEBUG)
Picamera2.set_logging(Picamera2.INFO)
# Examples: https://github.com/raspberrypi/picamera2/tree/main/examples
# Fake Long exposure: https://github.com/raspberrypi/picamera2/blob/main/examples/stack_raw.py

class Main(object):
    def __init__(self) -> None:
        self.configure = Configurator()
       
        self.res=get_res()
        self.configure.show()

class Viewfinder(QtWidgets.QMainWindow, Ui_Viewfinder):
    # Menu item properties
    menu_item_count = 17
    menu_item_count_exp = 29
    # Iso minimization properties
    been_minimized = False
    # HDR properties
    hdr_rel_exp = {0:0.5, 1:1, 2:1.5}
    HDR_counter = 0
    # Control properties
    custom_controls = { "AeEnable": False,  # Auto Exposure Value
                        "AwbEnable":False,  # Auto White Balance
                        "ExposureValue":0, # No exposure Val compensation --> Shouldnt be required as AeEnable:False
                        "NoiseReductionMode": controls.draft.NoiseReductionModeEnum.Off}
    zoom_index = 1 
    pixel_array = (4056, 3040) # TODO: Make method to retrieve

    def __init__(self):
        super().__init__()
        self.setupUi(self)
    
        self.style_sheet_stuff()
        
        # Allow toggle of gpio 4 - IR
        os.system('gpio -g mode 4 out')

        logging.info('Create Camera object')
        self.tuning = Picamera2.load_tuning_file(os.path.abspath(str(Path.home())+"/Camera/imx477_tuning_file_bare.json"))
        self.camera = Picamera2(tuning=self.tuning)
        
        if self.menu_item_count_exp == None:
            self.menu_item_count_exp = int(np.log2(self.camera.camera_controls['ExposureTime'][1]))
        # Set comboBox items camera_controls returns (min,max, current)
        # On second initiation this stops working so this
        self.ISO = (1,22,None)#self.camera.camera_controls['AnalogueGain']
        self.Exp = (114, 694422939, None)# self.camera.camera_controls['ExposureTime']
        ISO = self.ISO
        Exp = self.Exp
        for i in np.linspace(ISO[0],ISO[1],22): self.ISO_choice.addItem(str(i))
        for i in np.logspace(int(np.log2(114))+1,round(np.log2(694422939)),base=2,num = round(np.log2(694422939)) - int(np.log2(114))): self.exposure_choice.addItem(str(i))
        # FIXME: ISO an EXP none on load:
        self.custom_controls['AnalogueGain']=1
        self.custom_controls['ExposureTime']=1

        # ComboBox index Change
        self.ISO_choice.currentIndexChanged.connect(self.change_ISO)
        self.exposure_choice.currentIndexChanged.connect(self.change_exp)
        # Set Up camera
        logging.info('Setting up preview')
        self.set_preview()

        # Start Camera
        self.camera.start()
        logging.info('Camera Started')

    def style_sheet_stuff(self):
        # Fix up some of the UI parameters I couldnt figure out in QtDesigner 
        self.res = get_res()
        self.setFixedSize(*self.res)
        self.showFullScreen()

        # QWidget:
        self.Button_row.setGeometry(QtCore.QRect(int(self.res[0]*8.5/10), 0, int(self.res[0]*1.5/10), self.res[1]))
        #QVBoxLayout:
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(int(self.res[0]*8.5/10), 0, int(self.res[0]*8.5/10), self.res[1]))

        self.ZoomLabel.setText(str(self.pixel_array))

        # Resize main Layout widget
        self.gridLayoutWidget.setGeometry(QtCore.QRect(-1, -1, self.res[0], self.res[1]))      

        # Interface functions
        self.Exit.clicked.connect(self.exit)
        self.Zoom_button.clicked.connect(self.zoom)
        self.IR_button.clicked.connect(self.IR)
        self.Capture_button.clicked.connect(self.on_capture_clicked)

        self.Zoom_button.setStyleSheet('QPushButton {background-color: #455a64; color: #00c853;font: bold 30px;}')
        self.ZoomLabel.setStyleSheet('QLabel {background-color: #455a64; color: #00c853;font: bold 30px;}')
        self.Capture_button.setStyleSheet('QPushButton {background-color: #455a64; color: #00c853;font: bold 30px;}')
        self.Exit.setStyleSheet('QPushButton {background-color: #455a64; color: #00c853;font: bold 30px;}')
        self.exposure_choice.setStyleSheet('QComboBox {background-color: #455a64; color: #00c853;font: bold 30px;}')
        self.ISO_choice.setStyleSheet('QComboBox {background-color: #455a64; color: #00c853;font: bold 30px;}')
        self.IR_button.setStyleSheet('QPushButton {background-color: #455a64; color: #00c853;font: bold 30px;}')
        self.HDR_check.setStyleSheet('QCheckBox {background-color: #455a64; color: #00c853;font: bold 30px;}')

        self.ISO_label.setStyleSheet('QLabel {background-color: #455a64; color: #00c853;font: bold 30px;}')
        self.exposure_label.setStyleSheet('QLabel {background-color: #455a64; color: #00c853;font: bold 30px;}')
        self.ZoomLabel.setStyleSheet('QLabel {background-color: #455a64; color: #00c853;font: bold 30px;}')
        # Height
        self.Zoom_button.setFixedHeight(50)
        self.ZoomLabel.setFixedHeight(50)
        self.Capture_button.setFixedHeight(50)
        self.Exit.setFixedHeight(50)
        self.exposure_choice.setFixedHeight(50)
        self.ISO_choice.setFixedHeight(50)
        self.HDR_check.setFixedHeight(50)

    #           Dropdowns
    def change_ISO(self,index):
        self.camera.set_controls({'AnalogueGain':int(float(self.ISO_choice.itemText(index)))})
        logging.info('ISO -> '+str(self.ISO_choice.itemText(index)))
        if int(float(self.ISO_choice.itemText(index))) <= int(float(self.ISO[1])):
            self.custom_controls['AnalogueGain']=int(float(self.ISO_choice.itemText(index)))
        else:
            logging.warning('Selected ISO Value to large')

    def change_exp(self,index):
        self.camera.set_controls({'ExposureTime':int(float(self.exposure_choice.itemText(index)))})
        logging.info('Exp -> '+str(float(self.exposure_choice.itemText(index))))
        if int(float(self.exposure_choice.itemText(index))) <= int(float(self.Exp[1])): #I dont know why but the change exp button gets triggered on startup with a value slightly larger than this
            self.custom_controls['ExposureTime']=int(float(self.exposure_choice.itemText(index)))
        else:
            logging.warning('Selected Exposure to large')
        
    #           Buttons
    @QtCore.pyqtSlot()
    def zoom(self):
        '''https://github.com/raspberrypi/picamera2/blob/main/examples/zoom.py'''
        zooms = (1,0.5,0.25,0.1)
        # This iterates through different sensor modes all of variables sizes
        self.Zoom_button.setEnabled(False)

        new_size = [int(i*zooms[self.zoom_index]) for i in self.pixel_array]
        self.zoom_index += 1
        if self.zoom_index == len(zooms): self.zoom_index =0
        offset = [int((r - s) // 2) for r, s in zip(self.pixel_array, new_size)]
        self.camera.set_controls({"ScalerCrop": (*offset, *new_size)})
        self.ZoomLabel.setText(str(new_size))
        self.Zoom_button.setEnabled(True)
        self.Zoom_button.setStyleSheet('QPushButton {background-color: #455a64; color: #00c853;font: bold 30px;}')
        logging.debug('Changed zoom')

    def IR(self):
        """Toggles Infrared"""
        os.system('gpio -g toggle 4')

    
    @QtCore.pyqtSlot()
    def on_capture_clicked(self):
        """"""
        # FIXME: Measure how analogue gian affects brightness as it is not actually ISO -> after make some table and choose halving intervals 
        # Minimize ISO if possible
        #if float(self.ISO_choice.currentText())>1 and not self.been_minimized:
        #    # Current
        #    current_exp = float(self.exposure_choice.currentText())
        #    current_iso = float(self.ISO_choice.currentText())
        #    # Aim
        #    iso_aim = 5
        #    if self.HDR_check.isChecked():
        #        exp_aim = current_exp * self.hdr_rel_exp[2] * 2**(current_iso - 1)
        #    else:
        #        exp_aim = current_exp * 2**(current_iso - iso_aim)
        #    # Check if aim within feasable range
        #    while True:
        #        if exp_aim < self.camera.camera_controls['ExposureTime'][1]:
        #            break
        #        else:
        #            # Increment ISO aim
        #            iso_aim += 1
        #            # Decrement exp aim
        #            exp_aim /= 2
        #    if self.HDR_check.isChecked():
        #        self.custom_controls['ExposureTime'] = int(exp_aim/1.5)
        #    else:
        #        self.custom_controls['ExposureTime'] = int(exp_aim)
        #    self.custom_controls['AnalogueGain'] = int(iso_aim)

        #    self.camera.stop()
        #    self.camera.set_controls(self.custom_controls)
        #    self.camera.start()
        #    self.been_minimized = True

        if not self.HDR_check.isChecked():
            logging.info('Starting Capture {}'.format(dt.datetime.now().strftime('%m/%d/%Y-%H:%M:%S')))
            self.Capture_button.setEnabled(False)
            self.Capture_button.setStyleSheet('QPushButton {background-color: #FF1744; color: #ff1744;font: bold 30px;}')
        
            cfg = self.camera.create_still_configuration(raw={})
            # Hope dng works
            self.fname = dt.datetime.now().strftime('%m%d%Y-%H:%M:%S')
            self.camera.switch_mode_and_capture_file(cfg, str(Path.home())+'/Images/{}.png'.format(self.fname),
                                                 signal_function=self.qpcamera.signal_done)
        else:
            # Take HDR image
            # Set one file name and then do _ hdr img nr
            self.fname = dt.datetime.now().strftime('%m%d%Y-%H:%M:%S')
            self.do_hdr()

    def do_hdr(self):
        # if counter reached 3
        if self.HDR_counter == 0:
            # Set exp and iso back to original 
            current_exp = float(self.exposure_choice.currentText())
            current_iso = float(self.ISO_choice.currentText())
            self.custom_controls['ExposureTime'] = int(current_exp)
            self.custom_controls['AnalogueGain'] = int(current_iso)
            self.camera.set_controls(self.custom_controls)
            self.Capture_button.setEnabled(False)
            self.Capture_button.setStyleSheet('QPushButton {background-color: #FF1744; color: #ff1744;font: bold 30px;}')
        logging.info('Starting HDR Capture {}'.format(dt.datetime.now().strftime('%m/%d/%Y-%H:%M:%S')))
        # Increment HDR counter and keep copy of old for file naming
        HDR_counter = self.HDR_counter
        self.HDR_counter += 1
        # dict with num: rel exp time
        
        cfg = self.camera.create_still_configuration(raw={})
        # Set exp for HDR shot 
        self.mod_controls = self.custom_controls.copy()
        self.mod_controls['ExposureTime']=int(self.custom_controls['ExposureTime']*self.hdr_rel_exp[HDR_counter])
        # TODO: What is quicker waiting for frames to have settings applied or restart cam
        self.camera.stop()
        self.camera.set_controls(self.mod_controls)
        self.camera.start()
        # Take image
        self.camera.switch_mode_and_capture_file(cfg, str(Path.home())+'/Images/{}_{}.png'.format(self.fname, HDR_counter),
                                                signal_function=self.qpcamera.signal_done)
        

    @QtCore.pyqtSlot()
    def capture_done(self,*args):
        if not self.HDR_check.isChecked(): # none HDR imaging chain
            logging.info('Waiting {}'.format(dt.datetime.now()))
            if len(args) > 0:
                res = self.camera.wait(*args)
            else: logging.warning('Job completed before capture done called')
            logging.info('captured {}'.format(dt.datetime.now()))
            self.Capture_button.setEnabled(True)
            self.Capture_button.setStyleSheet('QPushButton {background-color: #455a64; color: #00c853;font: bold 30px;}')
            logging.info('ready')
            # Add Exif Data in new thread as it takes a while FIXME: Make exif threaded
            #cmd = 'exiftool -Exposure={} -ISO={} -Lens={} -overwrite_original {}'.format(self.custom_controls['ExposureTime'],
            #                                                             self.custom_controls['AnalogueGain'],'"EO Ultra Compact Objective"',
            #                                                             str(Path.home())+'/Images/{}.png'.format(self.fname))
            cmd = ['exiftool', '-Exposure={}'.format(self.custom_controls['ExposureTime']), '-ISO={}'.format(self.custom_controls['AnalogueGain']), 
                   '-Lens={}'.format('EO_ULC'), '-overwrite_original', str(Path.home())+'/Images/{}.png'.format(self.fname)]
            #threading.Thread(target=subprocess.run, args=(cmd)).start()
            subprocess.run(cmd)
        
        else: # HDR imaging chain
            logging.info('Waiting {}'.format(dt.datetime.now()))
            if len(args) > 0:
                res = self.camera.wait(*args)
            else: logging.warning('Job completed before capture done called')
            logging.info('captured {} HDR {}'.format(dt.datetime.now(), self.HDR_counter))
            # Add Exif Data in new thread as it takes a while
            #cmd = 'exiftool -Exposure={} -ISO={} -Lens={} -overwrite_original {}'.format(self.mod_controls['ExposureTime'],
            #                                                             self.mod_controls['AnalogueGain'],'"EO Ultra Compact Objective"',
            #                                                             str(Path.home())+'/Images/{}_{}.png'.format(self.fname, self.HDR_counter-1))
            if not hasattr(self, 'mod_controls'): self.mod_controls = self.custom_controls # TODO: Remove when iso comp fixed
            cmd = ['exiftool', '-Exposure={}'.format(self.mod_controls['ExposureTime']), '-ISO={}'.format(self.mod_controls['AnalogueGain']), 
                   '-Lens={}'.format('EO_ULC'), '-overwrite_original', str(Path.home())+'/Images/{}_{}.png'.format(self.fname, self.HDR_counter-1)]
            #threading.Thread(target=subprocess.run, args=(cmd)).start()
            subprocess.run(cmd)
            if self.HDR_counter == 3:
                logging.info('Completed HDR image')
                self.Capture_button.setEnabled(True)
                self.Capture_button.setStyleSheet('QPushButton {background-color: #455a64; color: #00c853;font: bold 30px;}')
                self.HDR_counter = 0
                return
            else:
                self.do_hdr()
        self.been_minimized = False

    @QtCore.pyqtSlot()
    def exit(self):
        self.qpcamera.cleanup()
        self.camera.stop()
        self.camera.close()
        self.hide()
        del self.camera

    def set_preview(self,new_cam = False):
        """Initiate camera preview redirected to Preview widget"""
        if new_cam : self.camera = Picamera2()
        # cfg
        # Disable que to keep memory free,
        self.camera.configure(self.camera.create_preview_configuration(queue=False ,main={"size":self.res})) 
        self.camera.set_controls(self.custom_controls)
        # GUI
        self.qpcamera = QGlPicamera2(self.camera,width=self.res[0], height=self.res[1],keep_ar=False)# 
        self.qpcamera.done_signal.connect(self.capture_done)
        self.Preview.addWidget(self.qpcamera, 0,0,1,1)

        return None
    
# TODO Add microstepping control in endstop window, and a thing that indicates its doing shit
class Configurator(QtWidgets.QMainWindow, Ui_MainWindow):
    img_config = {'HDR': False, 'IR':False, 'motor_x':False,'motor_y':False, 'motor_z':False, 'IR_and_normal':False,'step_size':1,}
    img_dir = None
    grid = None
    gamepad = None
    gpio_pins = {'x': {'enable':17, 'ms1':27, 'ms2':22, 'ms3':10, 'dir':9, 'step':11}, 
                 'y':None, 
                 'z':None,
                 'IR':4}
    endstops = []
    # 1 step at 16 ms to mm travel conversion
    ms16step_mm = 0.00025

    camera_config = { "AeEnable": False,  # Auto Exposure Value
                        "AwbEnable":False,  # Auto White Balance
                        "ExposureValue":0, # No exposure Val compensation --> Shouldnt be required as AeEnable:False
                        "NoiseReductionMode": controls.draft.NoiseReductionModeEnum.Off,
                        'ExposureTime':2,
                        'AnalogueGain':1}
    def __init__(self,gpio_pins=None) -> None:
        super().__init__()
        self.setupUi(self)

        self.assign_button_function()
        # TODO : Populate dropdowns
        if gpio_pins is not None:
            self.gpio_pins = gpio_pins
        else:
            logging.info('Using default gpio pins')

        self.basic_conf_and_style()

        return
    
    def basic_conf_and_style(self):
        self.checkbox_x.setCheckState(True)

        # Populate step size (Exp and iso are populated when opening graphical)
        self.combobox_step.addItem(str(5)+' == '+ str(5*self.ms16step_mm*1e3)+chr(956)+'m')
        for i in np.linspace(10,150,15):
            self.combobox_step.addItem(str(i)+' == '+ str(i*self.ms16step_mm*1e3)+chr(956)+'m')

        for i in np.linspace(1,22,22): self.combobox_gain.addItem(str(i))
        for i in np.logspace(int(np.log2(114))+1,round(np.log2(694422939)),base=2,num = round(np.log2(694422939)) - int(np.log2(114))): self.combobox_exp.addItem(str(i))

        return

    def assign_button_function(self):
        # Buttons
        self.button_dslr.clicked.connect(self.launch_viewfinder)

        self.pushbutton_exit.clicked.connect(self.exit)

        self.button_auto.clicked.connect(self.start_imaging)

        self.pushbutton_find_endstops.clicked.connect(self.make_endstops)

        self.activate_gamepad.clicked.connect(self.set_gamepad)

        # Dropdowns
        self.combobox_exp.currentIndexChanged.connect(self.change_exp)

        self.combobox_gain.currentIndexChanged.connect(self.change_gain)

        self.combobox_step.currentIndexChanged.connect(self.change_step)

        return


    def set_gamepad(self):
        """Activates motor control with gamepad"""
        # This class starts a thread that constantly reads out the gamepad, if input is found the movement is done
        self.create_grid_controler()
        self.gamepad = XboxController(grid = self.grid)
        return


    # Change dropdown 
    def change_exp(self,index):
        self.camera_config['ExposureTime'] = int(float(self.combobox_exp.itemText(index)))
        return
    
    def change_gain(self,index):
        self.camera_config['AnalogueGain'] = int(float(self.combobox_gain.itemText(index)))
        return

    def change_step(self,index):
        self.img_config['step_size'] = int(float(self.combobox_step.itemText(index).split('==')[0].strip(' ')))
        return

    @QtCore.pyqtSlot()
    def make_endstops(self):
        self.create_grid_controler()
        if not hasattr(self,'endstop_window'):
            self.endstop_window = Endstop_Window(parent=self,gridcontroler=self.grid)
        else:
            self.endstop_window.start_camera()
        self.endstop_window.show()

    def create_grid_controler(self):
        print('Initiate motors and grid control')
        # Initiate Engines, and checking if it already exists
        if self.checkbox_x.isChecked() and not hasattr(self,'mx'):
            self.mx = Motor_Control(gpio_pins=self.gpio_pins['x'],dx=1/8) # FIXME:If change dx change in move of grid controler
        if self.checkbox_y.isChecked() and not hasattr(self,'y'):
            raise Exception('Not Implemented')
        if self.checkbox_z.isChecked() and not hasattr(self,'z'):
            raise Exception('Not Implemented')
        if self.grid is None:
            self.grid = Grid_Handler(motor_x=self.mx, motor_y=None, motor_z = None)
            # Check if old gridbounds exist
            if os.path.isfile('grid'):
                print('Loading Old Gridbounds')
                with open(str(Path.home())+'/grid','r') as f:
                    cont = f.read()
                pos, bounds = cont.split('\n')
                # Populate bounds
                self.grid.pos = [int(i) for i in pos.split(':')[1].split(',')]
                self.grid.gridbounds = [int(i) for i in bounds.split(':')[1].split(',')]
        return
    
    @QtCore.pyqtSlot()
    def start_imaging(self): # TODO: Add thing to adjust exposure if brightness too low (if below 50% increase exp so that mean 50%)
        """Starts automatic imaging chain"""

        # Make marker so that rsync knows when to stop copying
        os.system('echo "True" > {}'.format(os.path.abspath(str(Path.home())+"/imaging.txt")))
        # Set mode for pin 4 (IR) if it hasnt been set yet
        os.system('gpio -g mode {} out'.format(self.gpio_pins['IR']))

        if self.grid is None: # Add message notify-send didnt work prob wont with qt in fullscreen
            return
        elif self.grid.gridbounds == [0]*self.grid.n_motors:
            return
        # Hide window as PyQT will start to freeze - screw doing with window 
        self.hide()

        # Establish dir name and make img collection dir
        dirs = os.listdir(os.path.join(str(Path.home()),'Images'))
        dirs = [i for i in dirs if 'img' in i]
        self.img_dir = os.path.join(str(Path.home()),'Images','img_'+str(len(dirs)))
        if os.path.isdir(self.img_dir): # In case this already exists just do this
            self.img_dir = 'img_{}'.format(dt.datetime.now().strftime('%Y%m%d-%h%M'))
        os.mkdir(self.img_dir) 
        os.chdir(self.img_dir)
        print('Changed directory to {}'.format(self.img_dir))
        
        # Imging config
        if self.checkbox_IR.isChecked() and not self.checkbox_IR_an_normal.isChecked():
            self.img_config['IR'] = True
            self.IR_filter(True)
        elif self.checkbox_IR_an_normal.isChecked():
            self.img_config['IR_and_normal'] = True
            # In this case IR will be triggered after normal images
            self.IR_filter(False)
        else:
            self.IR_filter(False)
            pass
        # HDR
        if self.checkbox_HDR.isChecked():
            self.img_config['HDR'] = True

        print('Starting imaging with HDR={}, IR and Normal={}, Just IR {}'.format(self.img_config['HDR'],self.img_config['IR_and_normal'], self.img_config['IR']  ))
        self.start_camera()
        tot_grid = self.make_grid()
        # Set ms to 16
        self.grid.change_ms(1/16)
        
        # Adjust grid to minimum movement FIXME: make applicablkle to more than 1 motor
        # Check which end pos is closer to, 0 or endstop
        if tot_grid[0][-1] - self.grid.pos[0] < self.grid.pos[0]:
            tot_grid[0] = reversed(tot_grid[0])
        # Check image brightness
        self.adjust_exp()
        # Iterate over grid
        for i in tot_grid[0]: # FIXME below only works for 1D array
            print('Moving to {} / {:.6}mm'.format(i,i*0.0025/16))
            self.grid.move_to_coord([i])
            # Wait for image to stabilize
            time.sleep(0.5)
            self.make_image()
            print('Finished Imaging for position {}'.format([i]))
        print('-------------------------\nCompleted imaging routine\n\n')
        # Save the grid in case we want to restart
        print('Saving imaging grid')
        with open(str(Path.home())+'/grid','w') as f:
            f.write('pos:{}\n'.format(','.join([str(i) for i in self.grid.pos])))
            f.write('endpoint:{}'.format(','.join([str(i) for i in self.grid.gridbounds])))
            
        self.grid.disable_all()
        # Release Camera
        self.camera.stop()
        self.camera.close()
        del self.camera
        self.show()
        # TODO: At the end of the run check if first and last image have same area in focus to check for travel error
        
        # Make marker so that rsync knows when to stop copying
        os.system('echo "False" > {}'.format(os.path.abspath(str(Path.home())+"/imaging.txt")))
    
    def adjust_exp(self,brightness = 0.2):
        """It happened a lot that the images were underexposed so we will first check
        brightness -> mean of image/max val should be above this value
        Assumes nothing in focus 
        """
        while True:
            # Will be uint8
            arr = self.camera.capture_array()
            max_val = np.iinfo(arr.dtype).max
            if np.mean(arr)/max_val > brightness:
                break
            else:
                # Double exp time
                print('Brightness to low, increasing to: {}'.format(self.camera_config['ExposureTime'] *2))
                self.camera_config['ExposureTime']= self.camera_config['ExposureTime'] *2
                self.camera.set_controls(self.camera_config)
                # Wait a second
                time.sleep(1)
        print('Final Exp {}'.format(self.camera_config['ExposureTime']))
        print('Final ISO {}'.format(self.camera_config['AnalogueGain']))

    def make_grid(self):
        tot_grid = []
        for i in range(len(self.grid.motors)):
            if self.grid.motors[i] != None:
                # We create grid if motor connected
                ind = 0 
                grid = []
                while ind < self.grid.gridbounds[i]:
                    grid.append(ind)
                    ind += self.img_config['step_size']
                # Add max point if not already present
                if self.grid.gridbounds[i] not in grid: grid.append(self.grid.gridbounds[i])
                tot_grid.append(grid)
        # FIXME: Add meshgrid  for multiple motors
        print('grid:')
        print(tot_grid)
        np.meshgrid(*tot_grid)
        return tot_grid

    def start_camera(self):
        # Start Camera
        self.tuning = Picamera2.load_tuning_file(os.path.abspath(str(Path.home())+"/Camera/imx477_tuning_file_bare.json"))
        self.camera = Picamera2(tuning=self.tuning)
        self.camera.set_controls(self.camera_config)
        cfg = self.camera.create_still_configuration(raw={})
        self.camera.configure(cfg)
        self.camera.start()
        print('Configured camera')
        return

    def make_image(self):
        """If IR and turns on IR takes image and then does either one image or HDR if HDR is set"""
        filename = '{}'.format('_'.join([str(i) for i in self.grid.pos]))

        if self.img_config['IR_and_normal']:
            self.IR_filter(False)
            self.take_im(filename+'_IR.dng')
            self.IR_filter(True)
        if self.img_config['HDR']: 
            mod_controls = self.camera_config.copy()
            for i in [self.camera_config['ExposureTime']*i for i in (0.5,1.5,1)]:
                # Change exp time
                print(i)
                mod_controls['ExposureTime'] = int(i)
                self.camera.set_controls(mod_controls)
                time.sleep(1)
                self.take_im(filename+'_exp{}mus.dng'.format(i))
        else:
            self.take_im(filename+'_NoIR.dng')
        return


    def take_im(self,filename):
        """Quick utility to make image so that above less cluttered"""
        request = self.camera.capture_request(wait=True)
        request.save_dng(filename)
        request.release()
        del request
        time.sleep(1)
        return


    def IR_filter(self,state):
        """
        state --> Bool : True IR filter on, False IR filter off
        """
        if state: os.system('gpio -g write {} 1'.format(self.gpio_pins['IR']))
        else: os.system('gpio -g write {} 0'.format(self.gpio_pins['IR']))
        return
 
    @QtCore.pyqtSlot()
    def launch_viewfinder(self):
        """Launch viewfinder (window with preview and iso exp settings)"""
        self.viewfinder = Viewfinder()
        self.viewfinder.show()
        return

    @QtCore.pyqtSlot()
    def exit(self):
        if self.grid is not None:
            self.grid.disable_all()
        sys.exit(0)


class Endstop_Window(QtWidgets.QMainWindow, Ui_Endstop_window): # TODO: Add exit button and message for disabled vs enabled controler
    """
    Window to move motors and calibrate endstops
    
    Note that due to the way axis movement is handled z will not be controlled if x and or y is not present
    So motors must be populated in order x y z r otherwise assignment will not work correctly
    """
    own_step = 1
    # In case controler is available for controlling the grid
    minimal_config = False
    def __init__(self, parent,gridcontroler = None):
        super().__init__()
        # pass reference to parent window
        self.parent = parent
        self.ms16step_mm = parent.ms15step_mm

        if self.parent.gamepad != None:
            # Dont show vertical layout
            self.minimal_config = True
            # Start timer to update pos etc
            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.update_grid_pos)
            # Update every second
            self.timer.start(1000)

        self.grid = gridcontroler

        self.setupUi(self)

        self.assign_button_function()

        self.set_layout()

        self.start_camera()

        self.showFullScreen()

        os.system('gpio -g mode 4 out')
        # Disables IR filter
        os.system('gpio -g write 4 0')

        return

    def update_grid_pos(self):
        """Updates all entries in the top row"""
        _translate = QtCore.QCoreApplication.translate
        self.menu_pos_placeholder.setTitle(_translate("Endstop_window", str(self.grid.pos)))
        self.menu_total_move.setTitle(_translate("Endstop_window", str(self.grid.tot_move)))
        self.menuendpoint_placeholder.setTitle(_translate("Endstop_window", str(self.grid.gridbounds)))
        self.menuLast_Move.setTitle(_translate("Endstop_window", str(self.parent.gamepad.last)))
        return

    def assign_button_function(self):
        # Moves TODO: Make all move actions pyqt slots so that the video keeps updating during
        self.pushbutton_exit.clicked.connect(self.exit)
        move10 = partial(self.move, 10)
        self.move_10.clicked.connect(move10)
        move5 = partial(self.move, 5)
        self.move_5.clicked.connect(move5)
        self.move_own.clicked.connect(self.move_own_implementation)
        moveneg10 = partial(self.move, -10)
        self.move__10.clicked.connect(moveneg10)
        moveneg5 = partial(self.move, -5)
        self.move__5.clicked.connect(moveneg5)
        move_own_implementation_neg = partial(self.move_own_implementation, True)
        self.move__own.clicked.connect(move_own_implementation_neg)
        # end and zero points
        self.set_zero.clicked.connect(self.set_zeropoint)
        self.set_max.clicked.connect(self.set_maximum)
        # Step size
        self.combobox_step_size.currentIndexChanged.connect(self.set_stepsize)
        #exit
        self.pushbutton_exit.clicked.connect(self.exit)
        self.menuExit.actionexit = self.menuExit.addAction('Exit')
        self.menuExit.actionexit.triggered.connect(self.exit)


    def move_own_implementation(self,negative=False):
        if negative:
            self.move(0-self.own_step)
        else:
            self.move(self.own_step)


    def clear_item(self, item):
        """Copied from https://stackoverflow.com/questions/37564728/pyqt-how-to-remove-a-layout-from-a-layout
        for deleting vertical layout"""
        if hasattr(item, "layout"):
            if callable(item.layout):
                layout = item.layout()
        else:
            layout = None

        if hasattr(item, "widget"):
            if callable(item.widget):
                widget = item.widget()
        else:
            widget = None

        if widget:
            widget.setParent(None)
        elif layout:
            for i in reversed(range(layout.count())):
                self.clear_item(layout.itemAt(i))


    def set_layout(self):
        # Set widget width, couldnt find in designer
        if not self.minimal_config:
            for widget in self.verticalLayout.children():
                widget.setFixedWidth(20)
        else:
            self.clear_item(self.verticalLayout)

        # Add ISO 
        self.iso = {}
        for i in range(1,17):
            self.iso[i]=self.menuISO.addAction(str(i))
            change_iso = partial(self.change_iso, i)
            self.iso[i].triggered.connect(change_iso)

        # Add dropdown items
        # Populate motors
        if self.grid.x is not None:
            self.combobox_motor.addItem('x')
        if self.grid.y is not None:
            self.combobox_motor.addItem('y')
        if self.grid.z is not None:
            self.combobox_motor.addItem('z')
        #Populate step size
        self.combobox_step_size.addItem(str(1)+' == '+ str(1*self.ms16step_mm*1e3)+chr(956)+'m')
        for i in np.linspace(20,200,10):
            # chr == give character of number
            self.combobox_step_size.addItem(str(i)+' == '+ str(i*self.ms16step_mm*1e3)+chr(956)+'m')
        self.combobox_step_size.addItem(str(300.0)+' == '+ str(300*self.ms16step_mm*1e3)+chr(956)+'m')
        self.combobox_step_size.addItem(str(400.0)+' == '+ str(400*self.ms16step_mm*1e3)+chr(956)+'m')
        self.combobox_step_size.addItem(str(500.0)+' == '+ str(500*self.ms16step_mm*1e3)+chr(956)+'m')
        

        return
    
    def change_iso(self,val):
        self.camera.set_controls({'AnalogueGain':val})
    
    def start_camera(self):
        """Starts camera in generic preview with IR filter removed"""
        self.camera = Picamera2()
        self.camera.configure(self.camera.create_preview_configuration())
        self.camera.set_controls({'AeEnable':True})
        self.qpcamera = QGlPicamera2(self.camera) 
        self.gridLayout.addWidget(self.qpcamera, 0,0)
        self.camera.start()
        return

    def set_stepsize(self, index):
        self.own_step = int(float(self.combobox_step_size.itemText(index).split('==')[0].strip(' ')))
        return

    @QtCore.pyqtSlot()
    def move(self,steps=1):
        """Moves motor by x steps, updates menu items keeping track of current pos and total moves since initiation"""
        motor = self.combobox_motor.currentIndex()
        # Order of list is [xyzr] not all must be given but prior indeces must be
        steps = motor*[0]+[steps]
        self.grid.move_dist(steps)
        # Update pos and total move
        _translate = QtCore.QCoreApplication.translate
        self.menu_pos_placeholder.setTitle(_translate("Endstop_window", str(self.grid.pos)))
        self.menu_total_move.setTitle(_translate("Endstop_window", str(self.grid.tot_move)))
        return

    @QtCore.pyqtSlot()
    def set_zeropoint(self):
        self.grid.make_zeropoint(axis=self.combobox_motor.currentIndex())
        return
    
    @QtCore.pyqtSlot()
    def set_maximum(self):
        _translate = QtCore.QCoreApplication.translate
        self.grid.make_endstop(axis=self.combobox_motor.currentIndex())
        self.menuendpoint_placeholder.setTitle(_translate("Endstop_window", str(self.grid.gridbounds)))
        return

    @QtCore.pyqtSlot()
    def exit(self):
        # Remove camera reference
        self.gridLayout.removeWidget(self.qpcamera)
        self.camera.close()
        self.hide()
        return



class Grid_Handler:
    """
    Class to keep track of the imaging grid

    dir: 0 --> Towards camera

    """
    # The displacement is measured in ms so the grid is step size dependent
    # TODO: Make step to mm conversion : 1 step (no micro) == 0.0025 mm
    # Keep track of bounds
    n_motors = 3
    gridbounds = [0]*n_motors
    # Position in grid
    pos = [0]*n_motors
    tot_move = [0]*n_motors
    last_pos = [0]*n_motors
    dx = 1
    # Bool to check if zeropoint set
    zero_made = False
    def __init__(self,motor_x,motor_y=None,motor_z=None) -> None:
        # Implement File handler here
        # Implement end stop control
        # Implement depth map recording 
        # Possibly implement automatic rsync from raspi to pc but dont know if that will work well
        # ----- Could just set rsync to run every 5 minutes when raspberry images
        
        # Motors
        self.x = motor_x
        self.x.enable()
        self.dx = self.x.dx
        if motor_y is not None:
            self.y = motor_y
        else:
            self.y = None
        if motor_z is not None:
            self.z = motor_z
        else:
            self.z = None
        self.motors = [self.x, self.y, self.z]
        pass
    
    def reset_grid(self,axis=None):
        """Resets the grid"""
        if axis == None:
            self.zero_made = False
            self.endstop = [0]*self.n_motors
        return

    def make_zeropoint(self,axis):
        """
        Sets pos for axis to zero
        """
        # In case max point was set prior to zero point
        if self.gridbounds[axis] != 0:
            # Incerment by current pos so that max point remains unchanged
            self.gridbounds[axis] += self.pos[axis]
        self.pos[axis] = 0
        self.zero_made = True
        return

    def make_endstop(self, axis, end=None):
        """
        end : current position and distance from 0 point TODO: Establish if this is meant to be front or back
        axis : int describing axes x:0, y:1, z:2 
        """

        if end == None:
            self.gridbounds[axis] = self.pos[axis]
        else:
            self.gridbounds[axis] = end
        return 

    def move_dist(self,disp, adjust_ms=True): # FIXME: Why coordinate out of grid?
        """
        disp: [x,...] list with displacement steps to go to
                - Length doesnt matter will only do for provided axes (based on list index)
        adjust_ms : adjust step for current stepping relative to grid (which is relative to ms16)

        Coordinate 0 is the furthest from the camera at the very left bottom as seen when facing the camera
        dir 0 causes movement towards the camera
        """
        # Change disp to ms 16 equivalent
        conv = {1:16, 1/2:8, 1/4:4,1/8:2, 1/16:1}
        # Keep move command for current microstepping
        # Disp is from here on out a book keeping tool
        move = disp
        if adjust_ms:
            disp = [int(disp[i]*conv[self.motors[i].dx]) for i in range(len(disp))]
        
        # Check that all within bounds
        for i in range(len(disp)): 
            # Check that bounds were set - if this is pre setting bounds this is ignored
            if self.gridbounds[i] != 0 and self.zero_made:
                if self.gridbounds[i] < self.pos[i]+disp[i]:
                    notification('Coordinate out of Grid (gt gb)')
                    return
                elif self.pos[i]+disp[i] <0:
                    notification('Coordinate out of Grid (lt 0)')
                    return
        
        # First set direction
        for i in range(len(disp)):
            # If disp negative -> movement away from camera so gpio dir = high <--- this may be wrong
            # if -disp and True do
            if disp[i] < 0 and not self.motors[i].dir: 
                self.motors[i].toggle_dir()
            elif disp[i] > 0 and self.motors[i].dir: 
                self.motors[i].toggle_dir()
                
        # Save last state 
        self.last_pos = self.pos
        # Do movement
        for i in range(len(move)):
            # Do disp steps times step size (microstepping)
            count = 0 
            for j in range(abs(move[i])): 
                self.motors[i].step()
                count += 1
            print("moved {} steps".format(count))
            print('Same as expected: {}'.format(count == abs(disp[i])))
        # Save new pos and tot move
        for i in range(len(disp)):
            self.pos[i] = self.pos[i]+disp[i]
            self.tot_move[i] = self.tot_move[i]+disp[i]

    def change_ms(self,ms = None):
        """
        originally for gamepad controler to iterate through ms
        if ms is specified changes to that one
        """
        if ms is None:
            dx_curr = self.dx
            if dx_curr/2 < 1/16: dx_curr = 2
            for i in self.motors:
                if i is not None:
                    i.set_step_mode(dx_curr/2)
            self.dx = dx_curr/2
        else:
            for i in self.motors:
                if i is not None:
                    i.set_step_mode(ms)
            self.dx = ms

    def move_to_coord(self,coord):
        """
        coord: [x,...] list with coords to go to
        IMPORTANT: MS must be 1/16 for it to produce grid reliable results
        Intended to be used by imaging script
        """
        # Get coord difference
        disp = [coord[i]-self.pos[i] for i in range(len(coord))]
        print(disp)
        self.move_dist(disp, adjust_ms=False)
        return
    
    def disable_all(self):
        if self.x is not None:
            self.x.disable()
        if self.y is not None:
            self.y.disable()
        if self.z is not None:
            self.z.disable()
        return
        




class Motor_Control: 
    """
    Motor Controler for Sparkfun Big Easy Driver - single motor control
    Initiate a second instance for both motors 

    """
    # True == On , False == Off
    enabled = False
    # False = default dir = pin low movement towards camera; True = other dir = pin high  movement away from camera
    dir = False
    delay = 0.1
    def __init__(self, gpio_pins={'enable':17, 'ms1':27, 'ms2':22, 'ms3':10, 'dir':9, 'step':11} , 
                 dx=1/16):
        """
        gpio_pins : dict of pin name to gpio location
         keys required -> ms1 ms2 ms3 enable step dir
        dx : Step size
        """
        self.dx = dx
        # Set mode to follow BCM not layout numbering
        GPIO.setmode(GPIO.BCM)
        # Set output mode --> All pins are output
        for key in gpio_pins:
            GPIO.setup(gpio_pins[key], GPIO.OUT)
        self.gpio_pins = gpio_pins
        self.set_step_mode(dx)
        # Set step low (dont know if it is by default --> Probably tho)
        GPIO.setup(self.gpio_pins['step'], GPIO.LOW)
        self.enable()
        return

    def set_step_mode(self, step_size):
        """Set stepping mode"""
        step_dir = {1:[0,0,0], 1/2:[1,0,0], 1/4:[0,1,0], 1/8:[1,1,0], 1/16:[1,1,1]}
        pinout = step_dir[step_size]
        pinout = [GPIO.HIGH if i==1 else GPIO.LOW for i in pinout]
        GPIO.setup(self.gpio_pins['ms1'], pinout[0])
        GPIO.setup(self.gpio_pins['ms2'], pinout[1])
        GPIO.setup(self.gpio_pins['ms3'], pinout[2])
        self.dx = step_size

    def toggle_dir(self):
        """Change state of dir pin"""
        if self.dir:
            GPIO.setup(self.gpio_pins['dir'], GPIO.LOW)
            self.dir = False
        else:
            GPIO.setup(self.gpio_pins['dir'], GPIO.HIGH)
            self.dir = True
        return

    def disable(self):
        """disable"""
        GPIO.setup(self.gpio_pins['enable'], GPIO.LOW)
        self.enabled = False
            
        return
    
    def enable(self):
        """Enable"""
        GPIO.setup(self.gpio_pins['enable'], GPIO.HIGH)
        self.enabled = True

        return
    
    def step(self):
        """Step is triggered by pulling gpio low to high"""
        # Cause step
        GPIO.setup(self.gpio_pins['step'], GPIO.HIGH)
        time.sleep(self.delay/2)
        # Bakc to orig
        GPIO.setup(self.gpio_pins['step'], GPIO.LOW)
        time.sleep(self.delay/2)
        return


    def close(self):
        # First disable
        GPIO.setup(self.gpio_pins['enable'], GPIO.LOW)
        # Set all back to low
        for key in self.gpio_pins:
            GPIO.setup(self.gpio_pins[key], GPIO.LOW)
        return




class XboxController(object): # Add way to turn off
    """
    Copied from: https://stackoverflow.com/questions/46506850/how-can-i-get-input-from-an-xbox-one-controller-in-python
    
    The grid controls implemented work as follows:
    left joystick : left positive movement in grid, right negative movement in grid (in x) internal class name: LeftJoystickY
    Y : Changes ms to the next mode (1->1/2->1/4->1/8->1/16)
    start button (3 lines parrallel) : Disable/Enable gamepad input
    B : make zeropoint
    X : make endstop
    A : reset grid
    Right Trigger : Enable finecontrol (1 sec delay after each motor input)

    """
    MAX_TRIG_VAL = math.pow(2, 8)
    MAX_JOY_VAL = math.pow(2, 15)
    control_allowed = False
    # checks for last input timestamp
    timestamp = time.time()
    input_timeout = 60*1000 # seconds created race condition somewhere i think so it keeps triggering 
    fine_control = False
    last =''
    # Boolean to stop commands from being stacked on top of one another
    cmd_running = False
    def __init__(self,grid=None):

        self.LeftJoystickY = 0
        self.LeftJoystickX = 0
        self.RightJoystickY = 0
        self.RightJoystickX = 0
        self.LeftTrigger = 0
        self.RightTrigger = 0
        self.LeftBumper = 0
        self.RightBumper = 0
        self.A = 0
        self.X = 0
        self.Y = 0
        self.B = 0
        self.LeftThumb = 0
        self.RightThumb = 0
        self.Back = 0
        self.Start = 0
        self.LeftDPad = 0
        self.RightDPad = 0
        self.UpDPad = 0
        self.DownDPad = 0

        self._monitor_thread = threading.Thread(target=self._monitor_controller, args=())
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

        self.grid = grid


    def add_grid(self,grid):
        """Add grid controler after xbox thread created"""
        self.grid = grid
        return



    def read(self): # return the buttons/triggers that you care about in this methode
        lx = self.LeftJoystickX
        # Make tristate 
        ly = self.LeftJoystickY
        a = self.A # Corresponds to a
        x = self.X # corresponds to y
        y = self.Y # corresponds to x
        b = self.B # corresponds to b
        return [lx, ly, a,x,y,b,self.Start]

    def tristate(self,lx):
        """Takes trigger input formats it to -1,0,1"""
        if lx >= 0.5: lx = 1
        elif lx < 0.5 and lx>-0.5: lx = 0
        elif lx < -0.5: lx =-1
        return lx

    def _monitor_controller(self):
        print('Controller Activated')
        while True:
            # Get gamepad input
            self.get_input()
            # Do control
            if self.control_allowed and self.grid is not None: 
                # Timeout condition
                if time.time() - self.timestamp >= self.input_timeout:
                    print('Disabled Controler by timeout')
                    self.control_allowed = False
                # Disable gamepad
                elif self.Start == 1:
                    print('Disabled Controler by button')
                    self.control_allowed = False
                # Else do action
                else:
                    self.do_input()
            # Enable keypad
            elif self.Start == 1:
                print('Enabled Controler')
                self.control_allowed = True
                time.sleep(1)
            else:
                pass
            self.get_input()

    def get_input(self):
        """Records gamepad input"""
        events = get_gamepad()
        for event in events:
            if event.code == 'ABS_Y': # Minus so that up is positive
                self.LeftJoystickY =  self.tristate(- event.state / XboxController.MAX_JOY_VAL) # normalize between -1 and 1
            elif event.code == 'ABS_X':# Minus so that left is positive
                self.LeftJoystickX = self.tristate(- event.state / XboxController.MAX_JOY_VAL) # normalize between -1 and 1
            elif event.code == 'ABS_RY':
                self.RightJoystickY = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
            elif event.code == 'ABS_RX':
                self.RightJoystickX = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
            elif event.code == 'ABS_Z':
                self.LeftTrigger = event.state / XboxController.MAX_TRIG_VAL # normalize between 0 and 1
            elif event.code == 'ABS_RZ':
                # tristate will set to 1 if triggered, the normalization goes between 0 and 4
                self.RightTrigger = self.tristate(event.state / XboxController.MAX_TRIG_VAL) # normalize between 0 and 1
            elif event.code == 'BTN_TL':
                self.LeftBumper = event.state
            elif event.code == 'BTN_TR':
                self.RightBumper = event.state
            elif event.code == 'BTN_SOUTH':
                self.A = event.state
            elif event.code == 'BTN_NORTH':
                self.X = event.state #previously switched with X
            elif event.code == 'BTN_WEST':
                self.Y = event.state #previously switched with Y
            elif event.code == 'BTN_EAST':
                self.B = event.state
            elif event.code == 'BTN_THUMBL':
                self.LeftThumb = event.state
            elif event.code == 'BTN_THUMBR':
                self.RightThumb = event.state
            elif event.code == 'BTN_SELECT':
                self.Back = event.state
            elif event.code == 'BTN_START':
                self.Start = event.state
            elif event.code == 'BTN_TRIGGER_HAPPY1':
                self.LeftDPad = event.state
            elif event.code == 'BTN_TRIGGER_HAPPY2':
                self.RightDPad = event.state
            elif event.code == 'BTN_TRIGGER_HAPPY3':
                self.UpDPad = event.state
            elif event.code == 'BTN_TRIGGER_HAPPY4':
                self.DownDPad = event.state
        return

    def do_input(self):
        # bool in case nothing is done
        if not self.cmd_running:
            self.cmd_running = True
            nothing = False
            move_cmd = False
            # Motor x control:
            if self.LeftJoystickX == 1:
                self.grid.move_dist([1])
                print('Move dist 1')
                last = 'move +1'
                move_cmd = True
            elif self.LeftJoystickX == -1:
                self.grid.move_dist([-1])
                print('Move dist -1')
                last = 'move -1'
                move_cmd = True
            # Step size
            elif self.Y == 1:
                self.grid.change_ms()
                print('changed ms: ', self.grid.dx)
                ms_str = {1:'1',1/2:'1/2', 1/4:'1/4', 1/8:'1/8',1/16:'1/16'}
                last = 'Change ms {}'.format(ms_str[self.grid.dx])

            elif self.B == 1:# TODO: Find out how to do this for other axis
                self.grid.make_zeropoint(0)
                print('Made zeropoint')
                last = 'Made Zeropoint'

            elif self.X == 1:
                self.grid.make_endstop(0)
                print('Made endstop')
                last = 'Made Endpoint'

            elif self.A == 1:
                self.grid.zero_made = False
                self.grid.endstop = [0]
                print('reset grid')
                last = 'Reset Grid'

            elif self.RightTrigger ==1:
                self.fine_control= True
                last = 'Enabled Finecontrol'
                print('Enabled Finecontrol')

            else:
                nothing = True
            # In case something was done record timestamp
            if not nothing: 
                self.last = last
                self.timestamp = time.time()
                # If it wasnt a move command we wait a second so the button isnt double triggered
                if not move_cmd:
                    time.sleep(1)
                if move_cmd and self.fine_control:
                    time.sleep(1)
            self.cmd_running = False

        return









def notification(msg):
    os.system('notify-send "{}"'.format(msg))





def get_res():
    """Returns screen self.resolution"""
    res = subprocess.run('xrandr', capture_output=True, check=True)
    cond = False
    res = str(res.stdout).split('\\n')
    for i in res:
        if cond : 
            res =i.split('   ')[1].strip(' ')
            res = res.split('x')
            res = (int(res[0]),int(res[1]))
            #Assume viewing in landscape
            if res[0]<res[1]:
                res = (res[1],res[0])
            return res
        if 'connected' in i:
            cond = True
    # Default if cant be found
    return (720,480)



if __name__=='__main__':

    app = QtWidgets.QApplication(sys.argv)
    main = Main()
    sys.exit(app.exec_())




