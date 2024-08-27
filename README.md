# Automated Microscope for macro/micro photography

The original idea of this project was to generate very high resolution images of small subjects by imaging over a very large grid using a linear stage. The magnification in the current set up is 2x, however, the goal is to go up to 20x eventually. 

With the final goal of turning these images into very high resolution 3d mappings of the subjects. Allthough I have no idea how to properly place multiple cameras for this, a new camera placement solution will have to be found for this, currently im envisioning a 4 camera set up, hanging above the stage, 3 of which are offset from the center lens to image at 120 degree angles relative to one another with the focus points for all 4 cameras at the same location, so essentially the cameras will be at the corners of a pyramid, and at the center of one plane will be a 4th camera with the same point in focus, this assembly will then be attached to a linear stage and suspended in some fashion above the actual stage. However, there is a number of problems with actually constructing this and sourcing enough microscope lenses. So this is far future. 

The old set up looks like:

<img src="./imgs/SetUp.jpg" alt="Microscope" width="800"/>

The current set up looks like:
<img src="./imgs/SetUp2.jpg" alt="Microscope" width="800"/>


The raw output of the microscope had to be increased, there is an issue where on occasion some images will have a large black area, this causes one in focus section to be very different from all the others, so every location is now imaged twice. But there is also the issue of images being over and under exposed quite frequently, so I every location is now imaged with 3 different exposures. This generates per full set (for hdf5) 3600GB of data, so images are processed in real time by a network mounted device onto which all data is saved (connected via ethernet, speed equivalent to USB2.0 ssd). And this device then fuses the 6 images and saves them in the format the old imaging script used to provide images. 

The relevant scripts for this now are:

NewDataAquisition
--> LiveImaging.py

FinalDataProcessing
--> RealTimeFocus.jl

After imaging is completed the normal MainFocus.jl may be run to generate the focus stacks, if the system (but particularly IO as there are a lot of read write operations occuring, ie write 6 images and a text file load all of them write a new file delete the old files) isn't overloaded one could even run this in parallel with the live_processing flag enabled.  


### Current major issue

I implemented that every position be imaged six times, twice per exposure with three exposures to deal with images occasionally being darker. The mkr fusion algorithm considers local color standard deviation, contrast and well exposedness in the form of deviation from 0.5. Neighboring images to the black images are considered well exposed (0.5 +- 0.2) so I would assume that this is not an overflow. The weight matrix considers all three weights equivalently in the sense that they are all normalized to 1 however not 0 to 1. So this leads me to believe that the 6 images generating the 1 darker image all are darker (somewhere around 0.2), my current ideas for this are:
- Maybe I should scale 0 to 1 and increase the weight matrix epsilon
- Maybe its LED flickering and some images just happen to be timed in such a way that the majority of images are taken during off time 
- Darkening appears to be somewhat symmetric beign darkest in image center and edges being less affected (might jsut be the ones im looking at rn), but I can see it being localized


Seocnd, the CCM is massively off, remeasure this very soon
Allthough I feel CCM is an underdetermined problem, in the sense that the 3x3 matrix is incapable of accurately correcting for all colors regardless of how well it is made. 
What could be feasable is making a custom CCM for each image by selecting for very similar colors and imaging them before and after the run. 

## Image Aquisition


This folder holds the code to control the microscope, a GUI frontend was written to control the camera exposure properties, and control the motors and a virtual grid. 

This code may no longer be functional as several untested changes have been made and a significant portion of the code is no longer required.

The main code for image acquisition is now in NewDataAquistion, utilizing scripts instead of a GUI as the GUI added way to much boilerplate code and complicated debugging and changing things quite significantly 

Also after ages of constantly messing up the computation because of stupitidy, I finally managed to compute the translation distance per step, it is 122 nm/step of the stepper motor. 


## Data Processing

Most files are older stages of the pipeline trialing different things. A working version of the Software will be provided in the subfolder Final Data Processing. 

Image stacking is now fully implemented with only two things left to do:
- RAM usage control, for a large number of x images the code gets killed due to using too much ram, this is solved for now by simply splitting it into 4 batches such that MKR runs 5 times, currently its set to 8 images at a time as I am still running this on my main PC and like to use it at the same time. 
- For some reason width and height of the image in the meta data supplied to the code is inverted, I think this is due to the camera being rotated in older setups and I fergot that while writing the code and testing alongside, so any width or height written may not be trusted, I wanted to fix this, but it leads to a lot of things breaking so I will adopt a more agnostic naming convention from now on, ie axes_1, axes_2 instead of width and height where applicable. 


## Taken Images

in the imgs folder one finds the images used in the markdown files and the images that have been taken

There is one ladybug image, which was combined from 8 individual images, and then combined by hand. 
This was taken with the very first iteration of the design where the camera and stage were detached leading to an incredibly frustrating alignment experience. 

The second is a bumbelbee this image sampled the entire imaging space and corresponds to a 18x15.7 k image, it is hand aligned as the MIST algorithm implementation wasnt complete at the time. And it was taken with the current set-up however, the surrounding velvet wasnt present yet so there was far more reflectance.

## Set-Up

The entire thing is controller by a raspberry CM4 on a CM-IO board, the camera corresponds to the raspberry HQ camera, the lens is an edmund optics UC objective, and the linear stage is something unbranded i got from ali express TODO: links

The linear stage has motors attached using 3d printed components to automate the movement, these correspond to 28BYJ-48 stepper motors with the standard driver one getts with them. At the maximal positions the individual motors are allowed to travel endstops are mounted.

I got a number of aluminium and steel pieces from some old hospital bed, these were used to attach the linear stage to a plate and mount the camera on top of it. The stepper drivers and CM4 are simply mounted to the plate using zipties and electric tape. 

Lighting is provided by a cheap ring light, its heating may be a source of extra noise, especially since I added a PVC (I think) seethrough tarp around the entire setup to keep dust out, its not great, but it does the job. 

The subject is elevated by a number of aluminium blocks the top one has some black velvet to minimize reflection into the camera, this might be adding extra reflection, ie a grey background, but hasn't been too bad, might redo the entire tarp casing for it in a while. 

## Future Set-up

Soon a server node from an old server will be made available to me, once I figure out how to make it quiet enough to keep on during image running all work will be offloaded to it, opening the possibilities of increasin magnification (ie currently each stack would produce way to much data to process in time due to storage limitations) and processing all data in real time. But a good enough lens (that doesnt increase magnification to the point that imaging would take even longer) must still be sourced.

There is also the 3d mapping plan, but that will require a mutlitude of changes for which I do not have the resources, expertise, and time at the moment. 