import requests
import rawpy
import numpy as np
import scipy
import os
import cv2 as cv
from PIL import Image
import json 
import skimage
import matplotlib.pyplot as plt

CUSTOM_DEBAYER = False

# Text colors
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def images_to_data(im_path,custom_debayer=CUSTOM_DEBAYER,extra=False):
    """
    Default is now to create effective pixels, if this limits resolution might be able
    to use debayering but im not sure its a good idea

    im_path : os path like object to the dng file
    custom_debayer : 2x2 sampling so one gets an effective pixel and an image with half dim as uint16
    extra : Return color array, color string and image without debayering as uint16
    """
    if '.png' not in im_path:
        # For raw formats
        im = rawpy.imread(im_path)
        if extra: # This part isnt in use anymore
            color_array = im.raw_colors
            im_data = im.raw_image_visible.copy()
            # The color array includes 0,1,2,3 with 3 colors the color string makes this a bit more obvious
            color_str = "".join([chr(im.color_desc[i]) for i in im.raw_pattern.flatten()])
            # 10 bit image to 16
            im_data = (im_data/(2**10-1)*(2**16-1)).astype(np.uint16)
            return color_array,color_str, im_data
        else: 
            # TODO: Look into demosaic algorithms 
            # 11:DHT best according to https://www.libraw.org/node/2306
            if not custom_debayer:
                return im.postprocess(demosaic_algorithm=rawpy.DemosaicAlgorithm(11),half_size=False, 
                                # 3color no brightness adjustment (default is False -> ie auto brightness)
                                four_color_rgb=False,no_auto_bright=True,
                                # If using dcb demosaicing
                                dcb_iterations=0, dcb_enhance=False, 
                                # Denoising
                                fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode(0),noise_thr=None,
                                # Color
                                median_filter_passes=0,use_camera_wb=False,use_auto_wb=False, user_wb=None,
                                # sRGB output and output bits per sample : 8 is default
                                output_color=rawpy.ColorSpace(1),output_bps=16,
                                # Black levels for the sensor are predefined and cant be modified i think
                                user_flip=None, user_black=None,
                                # Adjust maximum threshholds only applied if value is nonzero default was 0.75
                                # https://www.libraw.org/docs/API-datastruct.html
                                user_sat=None, auto_bright_thr=None, adjust_maximum_thr=0, bright=1.0,
                                # Ignore default is Clip
                                highlight_mode=rawpy.HighlightMode(1), 
                                # Exp shift 1 is do nothing, None should achieve the same but to be sure, preserve 1 is full preservation
                                exp_shift=1, exp_preserve_highlights=1.0,
                                # V_out = gamma[0]*V_in^gamma 
                                gamma=(1,1), chromatic_aberration=(1,1),bad_pixels_path=None
                                )
            else:
                # Resample colors so that each px = 2x2 px
                color_str = "".join([chr(im.color_desc[i]) for i in im.raw_pattern.flatten()])
                # 10 bit image
                color_array = im.raw_colors
                im = im.raw_image_visible.copy()
                h,w = im.shape
                # create standard rgb image
                arr = np.zeros((h//2,w//2,3))
                # Assign data
                arr[::,::,0] = im[color_array==0].reshape((h//2,w//2))
                arr[::,::,1] = (im[color_array==1]/2+im[color_array==3]/2).reshape((h//2,w//2))
                arr[::,::,2] = im[color_array==2].reshape((h//2,w//2))
                # Change to uint16 scale
                arr = (arr/(2**10-1)*(2**16-1)).astype(np.uint16)
                return arr
    else:
        return np.asarray(Image.open(im_path))
    
def im_to_dat_with_gamma(im_path, gamma= (2.222, 4.5) ):
    """Returns image data with gamma"""
    im = rawpy.imread(im_path)
    return im.postprocess(demosaic_algorithm=rawpy.DemosaicAlgorithm(11),half_size=False, 
                                # 3color no brightness adjustment (default is False -> ie auto brightness)
                                four_color_rgb=False,no_auto_bright=True,
                                # If using dcb demosaicing
                                dcb_iterations=0, dcb_enhance=False, 
                                # Denoising
                                fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode(0),noise_thr=None,
                                # Color
                                median_filter_passes=0,use_camera_wb=False,use_auto_wb=False, user_wb=None,
                                # sRGB output and output bits per sample : 8 is default
                                output_color=rawpy.ColorSpace(1),output_bps=16,
                                # Black levels for the sensor are predefined and cant be modified i think
                                user_flip=None, user_black=None,
                                # Adjust maximum threshholds only applied if value is nonzero default was 0.75
                                # https://www.libraw.org/docs/API-datastruct.html
                                user_sat=None, auto_bright_thr=None, adjust_maximum_thr=0, bright=1.0,
                                # Ignore default is Clip
                                highlight_mode=rawpy.HighlightMode(1), 
                                # Exp shift 1 is do nothing, None should achieve the same but to be sure, preserve 1 is full preservation
                                exp_shift=1, exp_preserve_highlights=1.0,
                                # V_out = gamma[0]*V_in^gamma 
                                gamma=gamma, chromatic_aberration=(1,1),bad_pixels_path=None
                                )
  

def master_bias(img_dir,out_dir):
    """Make master bias"""
    im = None
    count = 0 
    for i in os.listdir(img_dir):
        try:
            if im is None:
                im = images_to_data(os.path.join(img_dir,i),extra=False).astype(np.float64)
            else:
                im += images_to_data(os.path.join(img_dir,i),extra=False).astype(np.float64)
            count += 1
        except: pass
    master_bias = im/count
    np.save(os.path.join(out_dir,'master_bias.npy'), master_bias)


def master_flat(img_dir,out_dir, mbias,mdark):
    """Make master dark"""
    im = None
    count = 0
    for i in os.listdir(img_dir):
        try:
            if im is None:
                im = images_to_data(os.path.join(img_dir,i),extra=False).astype(np.float64)
            else:
                im += images_to_data(os.path.join(img_dir,i),extra=False).astype(np.float64)
            count += 1
        except: pass
    im /= count
    if type(mbias)!=np.ndarray and mbias!= None : 
        master_bias=np.load(mbias).astype(np.float64)
        im -= master_bias
    if type(mdark)!=np.ndarray and mdark!= None : 
        master_dark=np.load(mdark).astype(np.float64)
        im -= master_dark
    # Scale master flat between zero and 1
    for i in range(0,3):
        im[::,::,i]=im[::,::,i]/im[::,::,i].max()
    if im.min()<0.1:  print('Problem with flat images minimum value is to small (or dead pixels)')
    if np.sum(im==1)>im.size*0.01: print('Problem with flat images, to many pixels at maximum value')
    im = im.astype(np.float64)
    np.save(os.path.join(out_dir,'master_flat.npy'), im)


def compute_lens_correction(img_dir,save=True):
    """
    The general code required to compute the lens correction
    This should be run in a jupyter notebook to double check the results
    of the corner fitting algorithm
    Largely taken from 
    https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    and a lot of stackoverflow but i closed the links
    -----
    img_dir : path to directory with images of well lit subject containing a rectilinear grid
    out_dir : Where to save intermediate files
    save : wheter or not to save the generated mtx and dist data
    Any grid might actually work check the docs of cv calibrateCamera
    """ # TODO: Dtype scaling is wrong here fix at some point
    # Create averaged image we do this in here because we want rawpy to auto adjust brightness 
    # as the algorithm works better then 
    im = None
    count = 0 
    for i in os.listdir(img_dir):
        try:
            if im is None:
                im = images_to_data(os.path.join(img_dir,i)).astype(np.float64)
            else:
                im += images_to_data(os.path.join(img_dir,i)).astype(np.float64)
            count += 1
        except: pass
    lens_dist = im/count
   
    # switch back to standard image dtype for opencv
    if lens_dist.max() != 255: # TODO fix
        lens_dist = cv.convertScaleAbs(lens_dist, alpha=(255.0/65535.0)).astype(np.uint8)
    del im, count

    #           The next bit should prob be first test run in a notebook for new data
    # Cubes width height (approx to be varied to make it work)
    # Calibration goes completely wrong if the numbers are increased (second can go to 22 but prob best not to )
    cubes=(19,14)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((cubes[1]*cubes[0],3), np.float32)
    objp[:,:2] = np.mgrid[0:cubes[0],0:cubes[1]].T.reshape(-1,2)
    # First we change the image to a type accepted by opencv
    lens_dist = lens_dist.astype(np.uint8)

    # Now we change to grayscale and add a blur to make detection easier
    gray = cv.cvtColor(lens_dist, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray,11)
    blur = cv.GaussianBlur(gray, (3,3), 0, 0)
    # Perform median blur
    # Get rid of dead pixels, this requires the image to be properly exposed
    thresh = cv.adaptiveThreshold(blur, 255, 0, 1, 81, 15)
    #thresh = cv.adaptiveThreshold(blur, 255, 1, 1, 141, 16)
    # 31 corners per line 24 per column
    # detect corners with the goodFeaturesToTrack function.
    corners = cv.goodFeaturesToTrack(thresh, cubes[0]*cubes[1], 0.01, 55)
    corners = np.int0(corners)

    # Use the below to verify
    #for i in corners:
    #    x, y = i.ravel()
    #    cv.circle(lens_dist, (x, y), 5, 255, -1)
    #plt.imshow(lens_dist)
    # Refine the guess
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 1000, 0.001)
    corners2 = cv.cornerSubPix(thresh,np.ascontiguousarray(corners,dtype=np.float32), (15,15), (-1,-1), criteria)
    corners2 = corners2.astype(np.int64)
    

    # Again plot to check 
    #for i in corners2:
    #    x, y = i.ravel()
    #    cv.circle(thresh, (x, y), 5, 255, -1)
    #plt.imshow(thresh)

    # And now we get to the last bit , we get the calibration data
    #       Arrays are odd this works --> May require some messing with the nesting
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera([objp], [corners2.reshape((corners2.shape[0],2)).astype(np.float32)], gray.shape[::-1], None, None)
    # Dont know what ret is ->  its a sclar
    # mtx -> Camera matrix 
    # dist -> distance coefficience
    # rvecs, tvecs -> no clue 
    # dist is enough to map the lens correctly but you loose some information by using
    # mtx and dist in getOptimalNewCameraMatrix one apparently looses less information
    # TODO: Learn about the above and verify
    if save:
        with open(os.path.join(img_dir,'lens_calibration_data.json'),'w') as f:
            f.write(json.dumps({'mtx':mtx.tolist(),'dist':dist.tolist()}))
    else:
        return mtx,dist
    

def correct_lens(calibration,img):
    """
    calibration : os.path like or dictionary with the laoded data
    img : ndarray containing image data
    """
    if type(calibration)!=dict:
        with open(calibration,'r') as f:
            calibration=json.loads(f.read())

        for key in calibration:
            calibration[key]= np.array(calibration[key])

    h,  w = img.shape[:2]
    # This is required to keep the maximal amount of information ---> 
    # TODO: Figure out how to optimize alpha
    try:
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(calibration['mtx'], calibration['dist'], (w,h), 1, (w,h))
    except: # In case one forgets to change the dtype
        for key in calibration:
            calibration[key]= np.array(calibration[key])
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(calibration['mtx'], calibration['dist'], (w,h), 1, (w,h))
    dst = cv.undistort(img, calibration['mtx'], calibration['dist'], None, newcameramtx)
    return dst


def plotly_plot_img(img):
    """img : image as int dtype """
    import numpy as np
    import plotly.graph_objects as go
    import matplotlib.cm as cm

    # image dimensions (pixels)
    n1,n2,n3 = img.shape
    # Generate an image starting from a numerical function
    x, y = np.mgrid[0:n2:n2*1j,0:n1:n1*1j]
    fig = go.Figure(data=[
            go.Image(
                # Note that you can move the image around the screen
                # by setting appropriate values to x0, y0, dx, dy
                x0=x.min(),
                y0=y.min(),
                dx=(x.max() - x.min()) / n2,
                dy=(y.max() - y.min()) / n1,
                z=cv.convertScaleAbs(img, alpha=(255.0/np.iinfo(img.dtype).max))
            )
        ],
        layout={
            # set equal aspect ratio and axis labels
            "yaxis": {"scaleanchor": "x", "title": "y"},
            "xaxis": {"title": "x"}
        }
    )
    return fig