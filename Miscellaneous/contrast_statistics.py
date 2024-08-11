"""
Plots routines contrast statistics to select method for ImageFusion
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.optimize import curve_fit

dir = '/run/media/felix/3677421f-5daf-4ea7-ba33-31b09d11edcf/Images/img_0'

with open(os.path.join(dir, 'meta.txt'), 'r') as t:
    meta = t.readlines()
meta = [i.strip('\n').split(',') for i in meta]
meta = meta[1:]
meta = [[float(j) for j in i] for i in meta]

# Time since start, x, y, z, acceleration, contrast max min mean
try:
    fin_array = [[ i[3], i[0], i[1], i[2], i[4], i[5], i[6] ] for i in meta]
except:
    fin_array = [[ i[3], i[0], i[1], i[2], i[4], i[5], i[6] ] for i in meta[:-2]]

fin_array = np.array(fin_array)

orig_data = copy.deepcopy(fin_array)

# Make a plot of contrast values
sub = orig_data[:,5:]
plt.plot(np.abs(orig_data[:,4]), label='max')
plt.plot(np.abs(orig_data[:,5]), label='min')
plt.plot(np.abs(orig_data[:,6]), label='mean')

plt.legend()
plt.grid()
plt.savefig("Contrast Statistics")

plt.clf()

hist, bin_edges = np.histogram(orig_data[:,6], bins=100)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
cdf = np.cumsum(hist * np.diff(bin_edges))
cdf_normalized = cdf / cdf[-1]
# Plotting the histogram
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), color='blue', edgecolor='black', align='edge', label='Histogram')
ax1.set_xlabel('Contrast Statistics')
ax1.set_ylabel('Frequency', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Create secondary axis for the CDF
ax2 = ax1.twinx()
ax2.plot(bin_centers, cdf_normalized, 'r-o', label='CDF')
ax2.set_ylabel('Cumulative Distribution', color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.set_ylim(0, 1)  # Ensure the CDF ranges from 0 to 1

plt.title('Mean Contrast Statistics')
plt.legend()
ax1.grid(True, which='both')
#ax2.grid(True, which='both')
plt.savefig("Contrast Binning mean")
plt.clf()

hist, bin_edges = np.histogram(orig_data[:,4], bins=100)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
cdf = np.cumsum(hist * np.diff(bin_edges))
cdf_normalized = cdf / cdf[-1]
# Plotting the histogram
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), color='blue', edgecolor='black', align='edge', label='Histogram')
ax1.set_xlabel('Contrast Statistics')
ax1.set_ylabel('Frequency', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Create secondary axis for the CDF
ax2 = ax1.twinx()
ax2.plot(bin_centers, cdf_normalized, 'r-o', label='CDF')
ax2.set_ylabel('Cumulative Distribution', color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.set_ylim(0, 1)  # Ensure the CDF ranges from 0 to 1

plt.title('Max Contrast Statistics')
plt.legend()
ax1.grid(True, which='both')
#ax2.grid(True, which='both')
plt.savefig("Contrast Binning max")


def evaluate_thresholds(contrast_mean):
    contrast_mean = np.array(contrast_mean)  # Ensure it's a numpy array
    _mean = np.mean(contrast_mean)
    _median = np.median(contrast_mean)
    _std = np.std(contrast_mean)
    _min = np.min(contrast_mean)
    _max = np.max(contrast_mean)
    
    print("Computed Statistics:")
    print(f"{'Mean':<12}{'Median':<12}{'Std Dev':<12}{'Min':<12}{'Max':<12}")
    print(f"{_mean:<12.0f}{_median:<12.0f}{_std:<12.0f}{_min:<12.0f}{_max:<12.0f}\n")
    
    
    methods = {
        'mean - 0.5std': _mean - 0.5*_std,
        'median - std': _median - _std,
        'max - 2*std': _max - 2 * _std,
        'average(min, max) - std': (_min + _max) / 2 - _std,
        'average(mean, median) - std': (_mean + _median) / 2 - _std
    }
    
    print("Threshold Results:")
    print(f"{'Method':<30}{'Threshold':<15}{'% Above Threshold'}")
    for method, threshold in methods.items():
        percentage = np.mean(contrast_mean > threshold) * 100  # Percentage of values above the threshold
        print(f"{method:<30}{threshold:<15.0f}{percentage:.2f}%")


print("         Attemting methods on mean array")
evaluate_thresholds(orig_data[:,6])
print()
print("         Attemting methods on max array")
evaluate_thresholds(orig_data[:,4])
print()

"""
Mean and median are decently close 


"""
