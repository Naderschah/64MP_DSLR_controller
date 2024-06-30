# Automated Microscope for macro/micro photography

The original idea of this project was to generate very high resolution images of small subjects by imaging over a very large grid using a linear stage. The magnification in the current set up is 2x, however, the goal is to go up to 20x eventually. 

With the final goal of turning these images into very high resolution 3d mappings of the subjects.

The current set up looks like:

<img src="./imgs/SetUp.jpg" alt="Microscope" width="800"/>

## Image Aquisition


This folder holds the code to control the microscope, a GUI frontend was written to control the camera exposure properties, and control the motors and a virtual grid. 

This code may no longer be functional as several untested changes have been made and a significant portion of the code is no longer required.

A full rewrite of this is planned once the new setup will be realized. 


## Data Processing

Most files are older stages of the pipeline trialing different things. A working version of the Software will be provided in the subfolder Final Data Processing. 

While all files have some form of the working code they all aimed at adressing some problem.

The actually in use version of the code are found in FocusStackingMKR/Create_final_single_mkr.jl where the Marten Kautz van Reeth algorithm is implemented in julia and actually produces results. 

For image combining into a larger image, the MIST algorithm was partially implemented in julia, this can be found in MIST_reimplementation.jl


Both scripts work with some caveats, for the first one wants to reduce background in the image as much as possible, for the second some preprocessing kernels often help the algorithm align the images

I will be moving away from julia as I find all of the ways to organize my code to be rather aggitating, there is no nice way to group functions logically, for more info see [this](https://discourse.julialang.org/t/how-to-structure-project-in-julia/99458).


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

For the next iteration of the set up the camera will be top mounted, this is mainly since the extension tubes get quite long and ensuring alignment is much easier when gravity is helping. 

To do this an old xyz 3d-printer will be utilized, the hotend will be replaced with the camera and then raspberry pi will be mounted onto the other side of the vertical assembly to keep the camera cable short. The linear stage will be mounted where the hot-end was onece.
To achieve less reflectance of the 3d printer surfaces (all of which tend to be shiny) everything that may reflect will be painted with something like Black4.0, a highly absorbant acrylic paint. 

The 3d printer frame will then allow to move the camera higher allowing the mounting of extension tubes, furthmore the 3d printers motherboard will remain in place to control the already present stepper motors, with the possibility of controller the linear stage from it as well by sending commands from the raspberry. However, the last part is not quite certain yet. 

Lights will be mounted along the length of the camera holder. 

Cooling is an issue i havent quite figured out yet. 

3d mapping is lost in this set up as rotations will not add much spacial information anymore, however, I do have a spare raspi camera I might side mount to make lower resolution 3d models. 

For this the software will be completely rehauled. 