"""
Contains functions to minimize image difference through lens distortion fitting
"""
import numpy as np
import cv2 as cv
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr
import copy
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from numba import jit

## Main Optimization function
@jit(forceobj=True, looplift=True) #  Might need False
def scipy_optim_func(imgs, rel_pos, coeffs):
    # Everything moved into sanity check to easily check regions
    region0, region1 = sanity_check(imgs, rel_pos, coeffs)
    # Normalize color ranges
    region0 = (region0 - region0.min(axis=(0,1)))/(region0.max(axis=(0,1))-region0.min(axis=(0,1)))
    region1 = (region1 - region1.min(axis=(0,1)))/(region1.max(axis=(0,1))-region1.min(axis=(0,1)))
    # Compute Metrics
    _ssim = ssim(region0, region1, data_range = max(region0.max(), region1.max())-min(region0.min(), region1.min()), channel_axis=-1)
    _gmsd = GMSD(region0, region1)
    nr0, nr1 = region0.mean(axis=-1), region1.mean(axis=-1)
    _ncc = NCC(nr0, nr1)
    _pearson = pearsonr(((nr0 - nr0.min())/(nr0.max()-nr0.min())).flatten(),( (nr1 - nr1.min())/(nr1.max()-nr1.min())).flatten())
    return _ssim, _gmsd, _ncc, _pearson.statistic

@jit(forceobj=True, looplift=True)
def GMSD(img1, img2):
    # Gradient image
    G1, G2 = np.abs(cv.Laplacian(img1,cv.CV_64F)), np.abs(cv.Laplacian(img2,cv.CV_64F))
    # Compute Similarity 
    eps = 1e-9
    S = (2*G1*G2+eps) / (G1**2 + G2**2+eps)
    #  And std is GMSD
    return S.std()

@jit(nopython=True)
def NCC(image1, image2):
    mean1, mean2 = np.mean(image1), np.mean(image2)
    numerator = np.sum((image1 - mean1) * (image2 - mean2))
    denominator = np.sqrt(np.sum((image1 - mean1)**2) * np.sum((image2 - mean2)**2))
    return numerator / denominator if denominator != 0 else 0

#@jit(forceobj=True, looplift=True) #  Might need False
def sanity_check(imgs, rel_pos, coeffs):
    # Generate camera matrix
    cameraMatrix, distCoeffs = generate_CameraMatrixParameters(coeffs)
    h, w = imgs.shape[1:3]
    new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w, h), 1, (w, h))

    # Calculate the map for remapping
    mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, distCoeffs, None, new_camera_matrix, (w, h), 5)
    # Remap the images 
    imgs_ = np.empty_like(imgs)
    for i in range(2):
        imgs_[i] = cv.remap(imgs[i], mapx, mapy, cv.INTER_LINEAR)

    r1,r2 = compute_region_coordinates((w,h), copy.deepcopy(rel_pos))
    
    # We first extract the image region
    region0 = imgs_[0][r1[0]:r1[1],r1[2]:r1[3]]

    # Image 2 is offset by rel_pos, so we need to subtract this from the bounds to get the correct region
    region1 = imgs_[1][r2[0]:r2[1],r2[2]:r2[3]]

    # Remove black areas introduced from warping 
    region0,region1 = remove_black_borders(region0,region1)
    return region0, region1


def generate_image_sets():
    """
    Function to generate image sets, this has to be manually modified
    """
    # Generate array to hold all the shifts
    # num y is num row, num z is num col
    # giving the position as [y1,x1,y2,x2] where x1,y1 refer to the image to the left and x2y2 to the image on top
    rel_shifts = np.array(
        [   # idcs as y,z
            # 19660,     17914,           13435         8957          4478           0
            [ [0,0,0,0], [-6,137, 0,0] ,  [6, 320,0,0], [-4,335,0,0], [6,320, 0,0], [-4,328,0,0] ], # y=0
            # 19660,     17914,            13435         8957          4478           0
            [ [0,0,449,9], [-1,121,453,-8],[2,336,449,10],[0,318,452,-8],[3,336,448,9], [1,310,454,-8] ], # 6196
        ]
    )
    path = '/home/felix/RapidStorage2/LensDistortion'
    files = os.listdir(path)
    grid = [i.split("_") for i in files if os.path.isfile(os.path.join(path, i))]
    #               y,                      z             
    grid = np.array([[int(i[1].split("=")[1]), int(i[2].split("=")[1])] for i in grid])
    ys = np.sort(np.unique(grid[:,0]))
    zs = np.sort(np.unique(grid[:,1]))[::-1]
    # Generate all possible combinations, limit to max y investigated
    ys = ys[0:rel_shifts.shape[0]]
    image_sets = []
    rel_shifts_search = []
    for i in range(len(ys)):
        for j in range(len(zs)):
            if rel_shifts[i,j, 0] != 0 and rel_shifts[i,j, 1] != 0:  
                # Generate left right combinations
                image_sets.append([[ys[i],zs[j-1]], [ys[i], zs[j]]])
                rel_shifts_search.append(rel_shifts[i,j, 0:2])
            if rel_shifts[i,j, 3] != 0 and rel_shifts[i,j, 2] != 0:  
                # Generate top bottom combinations
                image_sets.append([[ys[i-1],zs[j]], [ys[i], zs[j]]])
                rel_shifts_search.append(rel_shifts[i,j, 2:4])
    image_sets = [["Focused_y={}_z={}_e=32000.png".format(j[0], j[1]) for j in i] for i in image_sets]
    image_sets = np.array([np.array([np.array(Image.open(os.path.join(path, j))) for j in i ]) for i in image_sets])
    return image_sets, rel_shifts_search

# Data Analysis
def plot_heatmap(grid_, Jout, xlabel, ylabel, make_log_range=False,plusminus=[1,1],marklargest= True):
    """
    Plots a heatmap based on sampled grid values and function evaluations.

    Parameters:
    - grid: numpy array of shape (2, n, m) where grid[0] contains x values and grid[1] contains y values
    - Jout: numpy array of shape (n, m) containing function evaluations at each grid point
    - xlabel: string for labeling the x-axis
    - ylabel: string for labeling the y-axis
    - make_log_range : When parameters are made logarithmic for investigating scales
    - plusminus : list of +- 1 depending on which quadrant was investigated for log
    """
    if make_log_range:
        grid = np.empty_like(grid_).astype('float')
        grid[0] = np.power(10.0,grid_[0])*plusminus[0]
        grid[1] = np.power(10.0,grid_[1])*plusminus[1]
    else:
        grid = copy.deepcopy(grid_)

    x_values = grid[0, :, 0] 
    y_values = grid[1, 0, :] 
    # Print max value
    print("{} : {} , {} : {} , val: {}".format(xlabel,grid[0].flatten()[np.nanargmax(Jout)],ylabel, grid[1].flatten()[np.nanargmax(Jout)], Jout.flatten()[np.nanargmax(Jout)]))
    # Overwrite main to max
    if marklargest:
        Jout[np.unravel_index(np.nanargmax(Jout), Jout.shape)] = 1

    # Create the heatmap
    plt.figure(figsize=(8, 6))
    heatmap = plt.imshow(Jout, extent=(x_values.min(), x_values.max(), y_values.min(), y_values.max()), origin='lower', aspect='auto', interpolation='none', cmap='viridis')
    plt.colorbar(heatmap)

    # Set the labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('Heatmap of Function Evaluations')

    # Set x and y ticks to reflect grid values
    plt.xticks(np.linspace(x_values.min(), x_values.max(), len(x_values)), labels=x_values)
    plt.yticks(np.linspace(y_values.min(), y_values.max(), len(y_values)), labels=y_values)

    # Show the plot
    plt.show()
# Appends to an axes
def plot_heatmap(grid_, Jout, xlabel, ylabel, title,make_log_range=False, plusminus=[1, 1], invert=False, marklargest=True, fig=None, ax=None):
    """
    Plots a heatmap based on sampled grid values and function evaluations.

    Parameters:
    - grid_: numpy array of shape (2, n, m) where grid_[0] contains x values and grid_[1] contains y values
    - Jout: numpy array of shape (n, m) containing function evaluations at each grid point
    - xlabel: string for labeling the x-axis
    - ylabel: string for labeling the y-axis
    - make_log_range: When parameters are made logarithmic for investigating scales
    - plusminus: list of +- 1 depending on which quadrant was investigated for log
    - marklargest: boolean indicating if the largest value should be highlighted
    - fig: matplotlib Figure object to which the heatmap will be added (optional)
    - ax: matplotlib Axes object where the heatmap will be plotted (optional)
    """
    if make_log_range:
        grid = np.empty_like(grid_).astype('float')
        grid[0] = np.power(10.0, grid_[0]) * plusminus[0]
        grid[1] = np.power(10.0, grid_[1]) * plusminus[1]
    else:
        grid = copy.deepcopy(grid_)
    if not invert:
        idx = np.nanargmax(Jout)
        print("{:10} | {} : {:7} , {} : {:7} , val: {}".format(title.split(" ")[0],xlabel,grid[0].flatten()[idx],ylabel, grid[1].flatten()[idx], Jout.flatten()[idx]))
    else:
        idx = np.nanargmin(np.abs(Jout))
        print("{:10} | {} : {:7} , {} : {:7} , val: {}".format(title.split(" ")[0],xlabel,grid[0].flatten()[idx],ylabel, grid[1].flatten()[idx], Jout.flatten()[idx]))


    x_values = grid[0, :, 0]
    y_values = grid[1, 0, :]
    cmap = 'viridis'
    if invert:
        cmap += '_r'
        vmin = Jout.min()
        vmax = Jout.mean()
    else:
        vmin = Jout.mean()
        vmax = Jout.max()
    # If no Axes object is passed, create a new figure and axes
    if ax is None:
        if fig is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            ax = fig.add_subplot(111)

    # Create the heatmap
    
    heatmap = ax.imshow(Jout,vmin=vmin,vmax=vmax, extent=(x_values.min(), x_values.max(), y_values.min(), y_values.max()), origin='lower', aspect='auto', interpolation='none', cmap=cmap)
    cbar = fig.colorbar(heatmap, ax=ax)

    if marklargest:
        if not invert:
            max_index = np.nanargmax(Jout)
        else: 
            max_index = np.nanargmin(np.abs(Jout))
        y,x = np.unravel_index(max_index, Jout.shape)
        ax.scatter(grid[0,x,y],grid[1,x,y],c='red')

    formatter = ScalarFormatter(useOffset=False)  # Creates a formatter that does not use scientific notation
    formatter.set_scientific(False)  # Ensure that the formatter doesn't switch to scientific notation
    cbar.formatter = formatter
    cbar.update_ticks()

    # Set the labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Set x and y ticks to reflect grid values
    ax.set_xticks(np.linspace(x_values.min(), x_values.max(), len(x_values)), labels=x_values)
    ax.set_yticks(np.linspace(y_values.min(), y_values.max(), len(y_values)), labels=y_values)
    formatter = FormatStrFormatter('%.0e')
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    # Return the Axes object
    return ax



@jit(forceobj=True, looplift=True)
def generate_CameraMatrixParameters(coeffs, focl_lenght_pxs = None):
    """
    coeffs : xc,yc, K1,K2,K3,P1,P2
    
    """
    if focl_lenght_pxs is None:
        focl_lenght_pxs = np.array([8, 8]) * np.array([1520, 2028]) / np.array([7.564,5.467])
    fx, fy = focl_lenght_pxs
    xc,yc, K1,K2,K3,P1,P2, = coeffs
    P3 = 0
    cameraMatrix = np.array([[fx, 0, xc],[0, fy, yc], [0,0,1]])
    distCoeffs =  np.array([K1,K2, P1,P2, K3])
    return cameraMatrix, distCoeffs

@jit(nopython=True) #  Might need False
def compute_region_coordinates(shape, rel_pos):

    w, h = shape
    # Maximum to avoid edge comparison
    max_width = int(0.8*w)
    max_height = int(0.8*h)
    # Base coordinates
    x1, y1 = 0, 0
    x2, y2 = rel_pos[1], rel_pos[0]
    udist_h, udist_w = h,w

    # Calculate the coordinates for the overlapping region
    left = max(x1,x2) 
    right = min(x1+udist_w, x2+udist_w) 
    top = max(y1, y2) # grid rel to top left with min value in top left
    bottom = min(y1+udist_h,y2+udist_h) 

    # We want symmetric indexing over the region center
    region_center_w = (left+right) // 2
    region_center_h = (top+bottom) // 2

    # We compute the maximum half width
    #                   First the range computed for left and right, then the maximum allowed, then the maximum possible for region 0 and region 1
    half_width = min(max_width//2, (udist_w - rel_pos[1])//2)
    half_height = min(max_height//2, (udist_h - rel_pos[0])//2)

    # We compute the new bounds of the region
    new_left = region_center_w - half_width
    new_right = region_center_w + half_width
    new_top = region_center_h - half_height
    new_bottom = region_center_h + half_height
    # R1 bounds
    r1 = [new_top,new_bottom,new_left,new_right]
    # R2 bounds
    r2 = [new_top - rel_pos[0] , new_bottom - rel_pos[0], new_left - rel_pos[1] , new_right - rel_pos[1]]
    return r1, r2

@jit(forceobj=True, looplift=True) #  Might need False
def remove_black_borders(imageA, imageB):
    """
    Required to compute ssim accurately

    Remove fully black rows and columns from both images.
    A pixel is considered 'black' if all its channels have a value of 0.
    """
    # Check for black rows and columns in both images
    # This assumes the images are grayscale or binary. If color, adjust the condition accordingly.
    if len(imageA.shape) == 3 and imageA.shape[2] == 3:
        non_black_rows_A = np.any(imageA.sum(axis=2) > 0, axis=1)
        non_black_cols_A = np.any(imageA.sum(axis=2) > 0, axis=0)
        non_black_rows_B = np.any(imageB.sum(axis=2) > 0, axis=1)
        non_black_cols_B = np.any(imageB.sum(axis=2) > 0, axis=0)
    else:
        non_black_rows_A = np.any(imageA > 0, axis=1)
        non_black_cols_A = np.any(imageA > 0, axis=0)
        non_black_rows_B = np.any(imageB > 0, axis=1)
        non_black_cols_B = np.any(imageB > 0, axis=0)


    # Combine results to keep rows and columns where either image has non-black data
    non_black_rows = non_black_rows_A | non_black_rows_B
    non_black_cols = non_black_cols_A | non_black_cols_B

    # Index and return the cropped images
    imageA_cropped = imageA[non_black_rows][:, non_black_cols]
    imageB_cropped = imageB[non_black_rows][:, non_black_cols]
    
    return imageA_cropped, imageB_cropped