#!/usr/bin/python
import gimpfu as gfu
import os

def place_images(path):
    # Parse directory
    y_val = []
    z_val = []# Focused_y=0_z=0_e=32000.png
    e_val = []
    for i in os.listdir(path):
        if not os.path.isdir(os.path.join(path, i)):
            _, y, z, e = i.split("_")
            y = y.split("=")[1]
            z = z.split("=")[1]
            e = e.split("=")[1].split(".")[0]
            if y not in y_val: y_val.append(y)
            if z not in z_val: z_val.append(y)
            if e not in e_val: e_val.append(y)

    y_val =sorted(y_val)
    z_val =sorted(z_val)
    e_val =sorted(e_val)
    e = e_val[0] # We only need 1 and the alginment grid will always be the same

    """
    Quick note on conventions
    <- z : increases to left
    |
    v
    y : increases downwards
    So we need to reorder z in computation
    """
    max_z = max(z_val)

    steps_to_px = 0.5/4096 / 2 / 1.55e-3 # (mm / step) / (mm/px) = px / step
    #y_val = map(lambda x: int(x*steps_to_px), y_val)
    #z_val = map(lambda x: int(x*steps_to_px), z_val)

    # Append steps traveled per image as proxy for image width, and add 200 to be sure
    width =  (max(y_val)+ y_val[1] - y_val[0]) * steps_to_px + 200
    height = (max(z_val)+ z_val[1] - z_val[0]) * steps_to_px + 200
    width = int(width)
    height = int(height)
    # Create new image holding all the subimages
    img = gfu.image(width, height, "RGB")
    # We now populate them all
    for y in y_val:
        for z in z_val:
            impath = os.path.join(path, "Focused_y={}_z={}_e={}.png".format(y, z, e))
            # Params: 0 : Run noninteractive, destination img, path
            layer = gfu.pdb["gimp-file-load-layer"](0, img, impath)
            # Img, Layer, parent_layer, layer position
            gfu.pdb["gimp-image-insert-layer"](img, layer, 0, -1)
            # Move the image -> With inverted z
            offx = int(abs(z-max_z) * steps_to_px)
            offy = int(y * steps_to_px)
            # layer, offx, offy -> Rel to img origin which is top left?
            gfu.pdb["gimp-layer-set-offsets"](layer, offx,offy)


gfu.register(
    "Place Image Grid",                 # name
    "Place Microscope Images on Grid",  # blurb
    "Place Microscope Images on Grid",  # help
    "Felix Semler",                     # author
    "Felix Semler",                     # copyright
    "2024",                             # Date
    "<Toolbox>/File/LoadMicroscopeGrid",# TODO: Menupath
    "",                                 # imagetypes
    [   # type, name, description, default
        (gfu.PF_STRING, "image_path", "Image Paths", ""), # Params
    ],
    [],                                 # Results
    place_images                        #function
)

gfu.main()