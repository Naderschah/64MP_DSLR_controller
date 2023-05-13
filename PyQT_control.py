
from UI_Viewfinder import Ui_Viewfinder
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

ZoomLevels = (1)


"""

TODOs: 
- Make sidebar QBoxLayout rather than widget
- SideBar background inconsisten color --> Make transparent
- centralWidget background --> make black
- Make buttons larger
- Add white balance sliders
- Figure out average imaging time 


qt5-tools designer
"""



# Setting CMA to 1024 doesnt work 512 does apparently CMA is required to come from the bottom 1GB of ram, which is shared with kernel gpu 


# Note all setting changes after camera start take a little to be enables (1-5s)
# For 64MP all sensor modes have a different crop limit

# Docs on NoiseReductionMode wrong (min:0 max:4 -> int) on site 3 options -> str : Assume 0 is no noise reduction
# WHat does Noise Reduction do?
logging.basicConfig(filename=str(Path.home())+'/Camera/logs/{}.log'.format(dt.datetime.now().strftime('%Y%m%d')), filemode='w', level=logging.DEBUG)
Picamera2.set_logging(Picamera2.DEBUG)
# Examples: https://github.com/raspberrypi/picamera2/tree/main/examples
# Fake Long exposure: https://github.com/raspberrypi/picamera2/blob/main/examples/stack_raw.py

class Main(object):
    def __init__(self) -> None:
        self.viewfinder = Viewfinder()
       
        self.res=get_res()
        self.viewfinder.show()

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
        self.ISO = self.camera.camera_controls['AnalogueGain']
        self.Exp = self.camera.camera_controls['ExposureTime']
        ISO = self.ISO
        Exp = self.Exp
        for i in np.linspace(ISO[0]-1,ISO[1],self.menu_item_count): self.ISO_choice.addItem(str(i))
        for i in np.logspace(start=1,stop=self.menu_item_count_exp,num=self.menu_item_count_exp,base=2): self.exposure_choice.addItem(str(i))
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
        
            cfg = self.camera.create_still_configuration()
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
        
        cfg = self.camera.create_still_configuration()
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
            # Add Exif Data in new thread as it takes a while
            cmd = 'exiftool -Exposure={} -ISO={} -Lens={} -overwrite_original {}'.format(self.custom_controls['ExposureTime'],
                                                                         self.custom_controls['AnalogueGain'],'"EO Ultra Compact Objective"',
                                                                         str(Path.home())+'/Images/{}.png'.format(self.fname))
            threading.Thread(target=subprocess.run, args=(cmd)).start()
        
        else: # HDR imaging chain
            logging.info('Waiting {}'.format(dt.datetime.now()))
            if len(args) > 0:
                res = self.camera.wait(*args)
            else: logging.warning('Job completed before capture done called')
            logging.info('captured {} HDR {}'.format(dt.datetime.now(), self.HDR_counter))
            # Add Exif Data in new thread as it takes a while
            cmd = 'exiftool -Exposure={} -ISO={} -Lens={} -overwrite_original {}'.format(self.mod_controls['ExposureTime'],
                                                                         self.mod_controls['AnalogueGain'],'"EO Ultra Compact Objective"',
                                                                         str(Path.home())+'/Images/{}_{}.png'.format(self.fname, self.HDR_counter-1))
            threading.Thread(target=subprocess.run, args=(cmd)).start()
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
        sys.exit(0)

    def set_preview(self,new_cam = False):
        """Initiate camera preview redirected to Preview widget"""
        # TODO: Check recommended for performance
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




