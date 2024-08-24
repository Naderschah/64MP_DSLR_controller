# Automated Microscope for macro/micro photography

The original idea of this project was to generate very high resolution images of small subjects by imaging over a very large grid using a linear stage. The magnification in the current set up is 2x, however, the goal is to go up to 20x eventually. 

With the final goal of turning these images into very high resolution 3d mappings of the subjects.

The old set up looks like:

<img src="./imgs/SetUp.jpg" alt="Microscope" width="800"/>

The current set up looks like:
TODO

Some general notes, my lens is limited to $4.44\mu m$, so the corresponding Nyquist sampling rate in magnification is $5.68x$ for a pixel size of $1.55\mu m$. 

There are a number of artifacts in the MKR stacked images especially for edges there is quite a few, this applies to any form of debayering, with and without ISP processing, so hopefully its spatial aliasing, allthough I doubt it. It is probably just a remnant from the algorithms reconstuction across image borders


## Image Aquisition


This folder holds the code to control the microscope, a GUI frontend was written to control the camera exposure properties, and control the motors and a virtual grid. 

This code may no longer be functional as several untested changes have been made and a significant portion of the code is no longer required.

The main code for image acquisition is now in NewDataAquistion, utilizing scripts instead of a GUI as the GUI added way to much boilerplate code and complicated debugging and changing things quite significantly 

Also after ages of constantly messing up the computation because of stupitidy, I finally managed to compute the translation distance per step, it is 122 nm/step of the stepper motor. 


## Data Processing

Most files are older stages of the pipeline trialing different things. A working version of the Software will be provided in the subfolder Final Data Processing. 

Image stacking is now fully implemented with only two things left to do:
- RAM usage control, for a large number of x images the code gets killed due to using too much ram, this is solved for now by simply splitting it into 4 batches such that MKR runs 5 times, a dynamic implementation of this (checking for total RAM used at maximum) is going to be added
- For some reason width and height of the image in the meta data supplied to the code is inverted, I think this is due to the camera being rotated in older setups and I fergot that while writing the code and testing alongside, this si to be changed but effectively only amounts to a renaming


## Taken Images

in the imgs folder one finds the images used in the markdown files and the images that have been taken

There is one ladybug image, which was combined from 8 individual images, and then combined by hand. 
This was taken with the very first iteration of the design where the camera and stage were detached leading to an incredibly frustrating alignment experience. 

The second is a bumbelbee this image sampled the entire imaging space and corresponds to a 18x15.7 k image, it is hand aligned as the MIST algorithm implementation wasnt complete at the time. And it was taken with the current set-up however, the surrounding velvet wasnt present yet so there was far more reflectance.

## Set-Up

The entire thing is controller by a raspberry CM4 on a CM-IO board, the camera corresponds to the raspberry HQ camera, the lens is an edmund optics UC objective, and the linear stage is something unbranded i got from ali express TODO: links

The linear stage has motors attached using 3d printed components to automate the movement, these correspond to 28BYJ-48 stepper motors with the standard driver one getts with them. At the maximal positions the individual motors are allowed to travel endstops are mounted.

The motor controllers and raspberry pi are housed in a box onto which the linear stage and camera are (poorly) attached. Cooling is provided by an attached fan mildly damped by a tpu insert. 

Lighting is provided by a hand made led array, there is a small one hugging the lens and a large one attached to the camera sensor, allthough the latter introduced quite some background noise, however, did reduce imaging time significantly. 

Around the entire set up one finds black velvet as it is pretty good at absorbing light and pretty cheap to source.


## Future Set-up

Outdated: TODO rewrite some time

For the next iteration of the set up the camera will be top mounted, this is mainly since the extension tubes get quite long and ensuring alignment is much easier when gravity is helping. 

To do this an old xyz 3d-printer will be utilized, the hotend will be replaced with the camera and then raspberry pi will be mounted onto the other side of the vertical assembly to keep the camera cable short. The linear stage will be mounted where the hot-end was onece.
To achieve less reflectance of the 3d printer surfaces (all of which tend to be shiny) everything that may reflect will be painted with something like Black4.0, a highly absorbant acrylic paint. 

The 3d printer frame will then allow to move the camera higher allowing the mounting of extension tubes, furthmore the 3d printers motherboard will remain in place to control the already present stepper motors, with the possibility of controller the linear stage from it as well by sending commands from the raspberry. However, the last part is not quite certain yet. 

Lights will be mounted along the length of the camera holder. 

Cooling is an issue i havent quite figured out yet. 

3d mapping is lost in this set up as rotations will not add much spacial information anymore, however, I do have a spare raspi camera I might side mount to make lower resolution 3d models. 

For this the software will be completely rehauled. 