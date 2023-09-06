#
# Here we generate HDR images using the approach (and code) presented in https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/PaperData/SIGGRAPH17_HDR.pdf
# https://github.com/TH3CHARLie/deep-high-dynamic-range/tree/master
# Kalantari, N.K., Ramamoorthi, R.: Deep High Dynamic Range Imaging of Dynamic Scenes. ACM TOG 36(4) (2017)
#
# The last nn checkpoint is copied and the code is reformated to produce outputs how I want them
# This method is optimized for correcting for movement as well as for hdr fusing images, this should help with alignment errors in my images, it may also be worth generating the hdr at the very end with this method
# Since image information fidelity is of interest we use the Weight Estimator network to not introduce any faulty information 
# For now use WIE though cause WE doesnt have any checkpoints

# The code has to be cloned into the directory this file is in to be used below

import sys
# Add code to search path
sys.path.insert(1, '/home/felix/PiCamera_DSLR_like_controller/DataProcessing/deep-high-dynamic-range')

from model import create_model_and_loss
import tensorflow as tf

# Generate WE mdoel 
model, _, output_function = create_model_and_loss('we')

# Retrieve checkpoint 
checkpoint = tf.train.Checkpoint(myModel=model)
checkpoint.restore(tf.train.latest_checkpoint("/home/felix/PiCamera_DSLR_like_controller/DataProcessing/deep-high-dynamic-range/saved-checkpoints/deepflow-wie"))

model.build(input_shape=(1, 912, 1412, 18))

# TODO : How to preprocessing --> and figure out later

# Generate HDR
outputs = model(inputs)