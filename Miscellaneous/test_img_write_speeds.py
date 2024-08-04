"""
test what datatypes are fastest to write to. 
Ie for png compression is done which takes up cpu time, but less data has to be written to disk


results:
HDF5 - Average Write Time: 0.703947 seconds, Std Dev: 0.570284
BMP - Average Write Time: 0.868212 seconds, Std Dev: 0.547201
PNG - Average Write Time: 18.203989 seconds, Std Dev: 0.007401

So can significantly speed this up by switching to HDF5

Also trial raw stream with manual debayer by overlapping, i.e. reduce nr of pixels by 4, so far have always been using main 

"""
import h5py
import numpy as np
import time
from PIL import Image
import statistics


def measure_time(method, *args, **kwargs):
    start_time = time.time()
    method(*args, **kwargs)
    return time.time() - start_time

# Number of trials
num_trials = 10

# Generate an image from the camera
from picamera2 import Picamera2, Preview
tuning = Picamera2.load_tuning_file("/usr/share/libcamera/ipa/rpi/vc4/imx477.json")
camera = Picamera2(tuning=tuning)
config = camera.create_still_configuration(raw={})
camera.start()
camera.switch_mode(config) 
camera.stop()

camera.set_controls({'AnalogueGain':1})
camera.set_controls({'ExposureTime':32000})
time.sleep(2)
camera.start()
image_data = camera.capture_array()
camera.stop()
print("Sensor modes")
print(camera.sensor_modes)
"""
Relevant output
          format unpacked  bit_depth          size             crop_limits
0  SRGGB10_CSI2P  SRGGB10         10   (1332, 990)  (696, 528, 2664, 1980)
1  SRGGB12_CSI2P  SRGGB12         12  (2028, 1080)    (0, 440, 4056, 2160)
2  SRGGB12_CSI2P  SRGGB12         12  (2028, 1520)      (0, 0, 4056, 3040)
3  SRGGB12_CSI2P  SRGGB12         12  (4056, 3040)      (0, 0, 4056, 3040)


"""
print("Configuration")
print(camera.camera_configuration()['raw'])
"""
Config set by simply setting config with raw={}

{'format': 'SBGGR12_CSI2P', 'size': (4056, 3040), 'stride': 6112, 'framesize': 18580480}
"""
"""
And the set config file
has color mode BGR888


Just noticed all formats have 8 bit per pixel per color
So for 12 need to use raw and debayer myself -> Could be nice to just reduce size by half to avoid any artifacts

"""


print("Img info")
print(image_data.shape)
print(image_data.dtype)
# HDF5 Write Times
hdf5_write_times = [
    measure_time(
        lambda: h5py.File(f'output{i}.h5', 'w').create_dataset("dataset", data=image_data).file.close()
    ) for i in range(num_trials)
]

# BMP Write Times
bmp_write_times = [
    measure_time(
        Image.fromarray(image_data).save, f'output{i}.bmp'
    ) for i in range(num_trials)
]

# PNG Write Times
png_write_times = [
    measure_time(
        Image.fromarray(image_data).save, f'output{i}.png'
    ) for i in range(num_trials)
]

# Calculate averages and standard deviations
hdf5_avg, hdf5_std = statistics.mean(hdf5_write_times), statistics.stdev(hdf5_write_times)
bmp_avg, bmp_std = statistics.mean(bmp_write_times), statistics.stdev(bmp_write_times)
png_avg, png_std = statistics.mean(png_write_times), statistics.stdev(png_write_times)

# Print out the results
print(f"HDF5 - Average Write Time: {hdf5_avg:.6f} seconds, Std Dev: {hdf5_std:.6f}")
print(f"BMP - Average Write Time: {bmp_avg:.6f} seconds, Std Dev: {bmp_std:.6f}")
print(f"PNG - Average Write Time: {png_avg:.6f} seconds, Std Dev: {png_std:.6f}")
