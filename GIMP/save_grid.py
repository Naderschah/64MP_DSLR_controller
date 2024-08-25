#!/usr/bin/python
import gimpfu as gfu
import os, csv

def save_grid(image, grid_path):
    if grid_path == "" or grid_path.split("/")[-1] != "":
        print("No path provided/Path invalid")
        return
    if ".csv" not in grid_path:
        grid_path += '.csv'
    grid = []
    for layer in image.layers:
        grid.append([layer.name, *layer.offsets])
    #layer.name #Holds file nam
    #layer.offsets # I think that holds the coordinates otherwise gpt says gfu.pdb.gimp_layer_get_offsets(layer)  but cant find it 
    with open(grid_path, 'w') as f:
        wr = csv.writer(f)
        for i in grid:
            wr.writerow(i)
    return


gfu.register(
    "Save Image Pixel Grid",            # name
    "Save Layer placement grid",        # blurb
    "Save Layer placement grid",        # help
    "Felix Semler",                     # author
    "Felix Semler",                     # copyright
    "2024",                             # Date
    "<Toolbox>/File/SaveMicroscopeGrid",# TODO: Menupath
    "",                                 # imagetypes
    [
        (gfu.PF_IMAGE, "image", "Input image", None),
        (gfu.PF_STRING, "grid_path", "Grid Path", ""), # Params
    ],
    [],                                 # Results
    save_grid                           #function
)

gfu.main()