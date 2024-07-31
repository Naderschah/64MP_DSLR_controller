## New SetUp

The code for the control of the microscope got quite messy. 

I will try to stay away from programming graphical applications to use it

The following scripts can be found here:

- FindEndstops.py
    - This will first ask the user to trigger each endstop manually to assure all function and are assigned to the correct axis (skippable by command line option)
    - Then it will make all axis move to their extremes and determine the grid endstops

- CheckCamera 
    - Takes images and opens them using feh to determine exposure and ISO required (no preview as way to slow over ssh)

- DoImaging 
    - Starts the imaging routine
    - Sources metadata from file
    - Has command line options for fraction of grid to observe - TODO: How are these most easily defined?

- CheckFocus 
    - Pre Imaging to allow centering of subject
    - Prints current gridbounds and mm conversion
    - Aligns camera to zero point of grid and closest to camera, and zero point and furthest from camera
    - TODO: Should I just use a monitor for this again? Moving back and forth is kinda annoying, but a monitor jsut for this is also kinda annoying

- Pinout.json
    - Contains the pinout for endstops stepper motors etc. 

- Grid_Classes.py 
    - Defines all objects used for this project

- Test_Endstops.py
    - First use function
    - To test if the endstop aligns to its mapping through Pinout.json

- Test_Motors.py
    - First use/utility function
    - Allows moving motors through simple commands
    - Usefull for determining desired movement direction and moving the stage to a different position


TODO: On grid off, move to start of step sequence to continue microstepping from there, record steps taken


