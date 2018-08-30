import numpy as np
import matplotlib.pyplot as plt
import time 
# display plots in this notebook
# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output r
import sys
caffe_root='/home/ch/caffe/'
sys.path.insert(0, caffe_root + 'python')

import caffe
caffe.set_mode_gpu()
model_def = '/home/ch/Alexnet/train_val_depth_deploy1.prototxt'
model_weights = '/home/ch/Alexnet/snapshot-depth/bvlc_iter_20000.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST) 
# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load('/home/ch/data/mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
image = caffe.io.load_image('/home/ch/Alexnet/1.jpg')
transformed_image = transformer.preprocess('data', image)
plt.imshow(image)
# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = transformed_image

### perform classification
a=time.time()
output = net.forward()
b=time.time()
print(b-a)
output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

print 'predicted class is:', output_prob.argmax()
