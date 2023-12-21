## Depth Stacking

In focus stacking the camera moves a little bit between images, on smaller magnifications this causes the position of objects in the image to change.

To solve this problem here an attempt is made to map the object in 3d space and then 2d project the resulting data, 

for starters the approach will be to apply simple geometry to solve for the positions in 3d space and do a direct 2d projection. 




Components:
- Geometry - Functions to solve the spacial coordinates of pixel positions
- InFocusMeasures - Functions to determine what pixels are in focus where
- Projections - Functions to 2d project the datacube

