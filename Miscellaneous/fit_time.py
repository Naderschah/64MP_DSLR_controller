"""
This script is dedicated to fitting recorded timing data of the microscope to provide better estimates

The function fitted has the form

Total = N x (exposure + readout) + movement + overhead


"""
import os
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.optimize import curve_fit

dir = '/run/media/felix/3677421f-5daf-4ea7-ba33-31b09d11edcf/Images/img_2'

with open(os.path.join(dir, 'timing.txt'), 'r') as t:
    timing = t.readlines()
print(timing)
timing = [float(i) for i in timing.split('\n')]

with open(os.path.join(dir, 'meta.txt'), 'r') as t:
    meta = t.readlines()

meta = [i.split(',') for i in meta.split('\n')]
meta = [[float(j.strip(':')) for j in i] for i in meta]

# Time since start, x, y, z, acceleration
fin_array = [[ timing[i], meta[i][0], meta[i][1], meta[i][2], meta[i][3] ] for i in range(len(timing))]

fin_array = np.array(fin_array)

orig_data = copy.deepcopy(fin_array)

# Subtract since last
fin_array[:,:-2] -= np.roll(fin_array, -1)[:,:-2]

# Remove edge points
fin_array = fin_array[1:-2]

# We now have:
# Time since last image, delta (x,y,z) , acceleration

times = fin_array[:, 0]
max_times = np.max(times)
absdist = np.abs(fin_array[:,1])+np.abs(fin_array[:,2])+np.abs(fin_array[:,3])
max_absdist = np.max(absdist)
accel = fin_array[:,-1]
max_accel = np.max(accel)

# Check if these are roughly constant throughout
plt.plot(np.linspace(len(times)), times/max_times, label='Max t: {}'.format(max_times))
plt.plot(np.linspace(len(times)), absdist/max_absdist, label='Max d: {}'.format(max_absdist))
plt.plot(np.linspace(len(times)), accel/max_accel, label='Max a: {}'.format(max_accel))

plt.legend()
plt.grid()
plt.title("Comparing variation of time since last image, absolute distance traveled, and acceleration over datapoints, all normalized to 1")
plt.savefig('Norm_comparison.png')


plt.clf()


# Now lets fit some functions
def time_estimate(exp,N,dist, readout, overhead):
    return N*(exp+readout) + (dist)/100 + overhead

def wrapped(x, readout,overhead):
    N, exp, dist = x
    return time_estimate(exp,N,dist, readout, overhead)

exp = np.array([48000 * 1e-6]*len(times))
N = np.array([1]*len(times))

# Fit using the one to one data
res1 = curve_fit(wrapped, (N, exp, absdist), times)

# Now fit using the entire data set
# Exp remains the same, N is now the index of the array + 1

N2 = np.arange(1, len(orig_data)+1)
times2 = orig_data[:, 0]
exp2 = np.array([48000 * 1e-6]*len(orig_data))
cumul_dist = [0] # First we assume has traveled 0

for i in range(len(orig_data)):
    if i != 0:
        cumul_dist.append(cumul_dist[i-1] + np.abs(np.abs(fin_array[i,1])+np.abs(fin_array[i,2])+np.abs(fin_array[i,3])-(np.abs(fin_array[i-1,1])+np.abs(fin_array[i-1,2])+np.abs(fin_array[i-1,3]))))

# We need to remove the first datapoint as it is not controlable where from the grid starts
N2 = N2[:-2] # Remove end since this is number of images
times2 = (times2 - times2[0])[1:] # Subtract first time recorded such that we effectively start from the second
cumul_dist = cumul_dist[1:] # Remove 0 entry
exp2 = exp2[1:] # Whichever

res2 = curve_fit(wrapped, (N2, exp2, cumul_dist), times2)

# and lastly we combine all of the arrays to see if that improves prediction

N = np.concatenate((N,N2),axis=0)
times = np.concatenate((times,times2),axis=0)
dist = np.concatenate((absdist, cumul_dist),axis=0)
exp = np.concatenate((exp,exp2))

res3 = curve_fit(wrapped, (N, exp, dist), times)



print("Results")
print("Only using N=1")
popt, pcov = res1
pred_time1 = wrapped((N, exp2, cumul_dist), popt[0], popt[1]) 
print("Popt: {}, with std: {}".format(popt,np.sqrt(np.diag(pcov))))
print("Only using continuous N")
popt, pcov = res2
pred_time2 = wrapped((N, exp2, cumul_dist), popt[0], popt[1]) 
print("Popt: {}, with std: {}".format(popt,np.sqrt(np.diag(pcov))))
print("Using Both")
popt, pcov = res3
pred_time3 = wrapped((N, exp2, cumul_dist), popt[0], popt[1]) 
print("Popt: {}, with std: {}".format(popt,np.sqrt(np.diag(pcov))))


# Now some more plotting for second set of data
plt.plot(N2, times2, label='time') 
plt.plot(N2, cumul_dist, label='distance') 

plt.plot(N2, pred_time1, label='Pred 1')
plt.plot(N2, pred_time2, label='Pred 2')
plt.plot(N2, pred_time3, label='Pred 3')

plt.legend()
plt.grid()
plt.savefig("TimePredictionsVsN")


# And again against cumulative distance
plt.plot(cumul_dist, times2, label='time') 
plt.plot(cumul_dist, N2, label='NumberImages') 

plt.plot(cumul_dist, pred_time1, label='Pred 1')
plt.plot(cumul_dist, pred_time2, label='Pred 2')
plt.plot(cumul_dist, pred_time3, label='Pred 3')

plt.legend()
plt.grid()
plt.savefig("TimePredictionsVsDist")

#And lastly some acceleration statistics
print("Acceleration statistics")
print("Mean: {}".format(np.mean(accel)))
print("Std: {}".format(np.std(accel)))
print("Min: {}".format(np.min(accel)))
print("Max: {}".format(np.max(accel)))