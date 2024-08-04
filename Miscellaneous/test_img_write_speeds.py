"""
test what datatypes are fastest to write to. 
Ie for png compression is done which takes up cpu time, but less data has to be written to disk


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
print("Configuration")
print(camera.camera_configuration()['raw'])

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
