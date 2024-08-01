
This directory contains all code regarding image processing. 

All files Main are for the full routine


MainFocus
- Runs focusing, note the max images parameter in the main function, 50 images took 1.5 hours per focus stack, 35 only 1 hour, we want to avoid moving images to swap, there will be an inflection point where it is favorable to load images into swap rather than parts from hard drive, find it
- Currently work is being done to prefilter for images on the raspberry pi based on contrast values

50 : 1.5hrs
35 : 1hrs 
20 : 0.5 hrs Still using a lot of swap
10 : ~0.5hrs But now (from htop) using max 60% mem vsz seems to max out at 25GB


MainCombine
- So far only CentralAlign has been tested in this implementation, MIST is to be done
- Fiji was a trial, not worth using