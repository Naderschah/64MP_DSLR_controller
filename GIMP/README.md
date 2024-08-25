## GIMP Assistance Scripts

If you have GIMPv3 and ScriptFu-v3 is still a thing, try the python scripts they arent tested but should be pretty close to working.

Otherwise the TinyScheme scripts (.scm) are tested and work. 

The basic idea is that load_images.scm reads a directory checks for all png's in this path, and then loads them into the image, it also offsets them using the conversion from the steps to px coordinates. 

The script save_images.scm loads for each layer, the layer name (which by construction corresponds to the filename) and their x,y coordinates relative to the image top left (or whichever way around gimp returns it havent checked yet) and saves them to a file in the given path called grid.txt. This can then be sourced in the image alignment scripts (such that hand alignment and complex fusion procedures are in scope) to avoid any coordinate computation within the scripts. 