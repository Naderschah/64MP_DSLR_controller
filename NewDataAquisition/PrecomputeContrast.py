import cv2
import timeit
from memory_profiler import profile
#pip install memory_profiler
# To get resutls:
# python -m memory_profiler example.py

"""
Results: 
For UInt8 of size (4056, 3040, 3)
Time per:
9 : 0.577
7 : 0.493
5 : 0.426
3 : 0.300
Memory:
9 : 90.4
7 : 90.4
5 : 90.4
3 : 90.7 # The fuck


So 9 is fine
"""


DEBUG = False
# Copied to DoImaging
def compute_contrast(image, kernel_size=9):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

    # Generate LoG kernel
    blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    # Change CV_32F to whatever the datatype of the passed or gray image is
    laplacian = cv2.Laplacian(blur, cv2.CV_32F, ksize=kernel_size)
    
    # Calculate max, min, and mean
    return laplacian.max(), laplacian.min(), laplacian.mean()








@profile()
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
        print("Running Time Tests")
        repeat = 3
        number = 100
        time_results = timeit.repeat("compute_contrast(img,9)", setup="from __main__ import compute_contrast\nimport cv2\nimg = cv2.imread('{}')".format(im_path), repeat=repeat, number=number)
        print(f"Average execution time: {sum(time_results)/len(time_results)/number} seconds")
        time_results = timeit.repeat("compute_contrast(img,7)", setup="from __main__ import compute_contrast\nimport cv2\nimg = cv2.imread('{}')".format(im_path), repeat=repeat, number=number)
        print(f"Average execution time: {sum(time_results)/len(time_results)/number} seconds")
        time_results = timeit.repeat("compute_contrast(img,5)", setup="from __main__ import compute_contrast\nimport cv2\nimg = cv2.imread('{}')".format(im_path), repeat=repeat, number=number)
        print(f"Average execution time: {sum(time_results)/len(time_results)/number} seconds")
        time_results = timeit.repeat("compute_contrast(img,3)", setup="from __main__ import compute_contrast\nimport cv2\nimg = cv2.imread('{}')".format(im_path), repeat=repeat, number=number)
        print(f"Average execution time: {sum(time_results)/len(time_results)/number} seconds")
    
    else: 
        compute_contrast(img,9)
    print("Memory profiling")
    res = mem_compute_contrast(img,9)
    res = mem_compute_contrast(img,7)
    res = mem_compute_contrast(img,5)
    res = mem_compute_contrast(img,3)



if __name__ == '__main__':
    im_path = '/home/micro/test_im.png'
    time_contrast(im_path)
