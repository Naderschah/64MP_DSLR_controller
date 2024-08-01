import cv2
import timeit
from memory_profiler import profile
#pip install memory_profiler
# To get resutls:
# python -m memory_profiler example.py


DEBUG = True

def compute_contrast(image, kernel_size=9):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

    # Generate LoG kernel
    blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    # Change CV_32F to whatever the datatype of the passed or gray image is
    laplacian = cv2.Laplacian(blur, cv2.CV_32F)
    
    # Calculate max, min, and mean
    return laplacian.max(), laplacian.min(), laplacian.mean()

@profile
def mem_compute_contrast(image,kernel_size=9):
    return compute_contrast(image, kernel_size=9)
    
def time_contrast(im_path, kernel_sizes = [3, 5, 7, 9]):
    img = cv2.imread(im_path)
    print('type(img)')
    print(type(img))
    print('img.shape')
    print(img.shape)
    print('img.dtype')
    print(img.dtype)
    print('img.size')
    print(img.size)

    if not DEBUG:
        time_results = timeit.repeat("compute_contrast({},9)".format(im_path), setup="from __main__ import process_image", repeat=5, number=1000)
        print(f"Average execution time: {sum(time_results)/len(time_results)} seconds")
        time_results = timeit.repeat("compute_contrast({},7)".format(im_path), setup="from __main__ import process_image", repeat=5, number=1000)
        print(f"Average execution time: {sum(time_results)/len(time_results)} seconds")
        time_results = timeit.repeat("compute_contrast({},5)".format(im_path), setup="from __main__ import process_image", repeat=5, number=1000)
        print(f"Average execution time: {sum(time_results)/len(time_results)} seconds")
        time_results = timeit.repeat("compute_contrast({},3)".format(im_path), setup="from __main__ import process_image", repeat=5, number=1000)
        print(f"Average execution time: {sum(time_results)/len(time_results)} seconds")
    
    else: 
        compute_contrast(im_path,9)
    print("Memory profiling")
    res = mem_compute_contrast(9)
    res = mem_compute_contrast(7)
    res = mem_compute_contrast(5)
    res = mem_compute_contrast(3)

    
    
