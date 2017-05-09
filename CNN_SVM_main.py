# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import caffe
# GPU_ID = 0 # Switch between 0 and 1 depending on the GPU you want to use.
# caffe.set_mode_gpu()
# caffe.set_device(GPU_ID)
from datetime import datetime
import numpy as np
import sys, getopt
import cv2
import os

from caffe.io import array_to_blobproto
from collections import defaultdict
from skimage import io

from compute_mean import compute_mean 




def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data); plt.axis('off')

	cv2.waitKey(0)






def main(argv):
	
	model_filename = ''
	weight_filename = ''
	img_filename = ''
	try:
		opts, args = getopt.getopt(argv, "hm:w:i:")
		print opts
	except getopt.GetoptError:
		print 'CNN_SVM_main.py -m <model_file> -w <output_file> -i <img_folder>'
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print 'CNN_SVM_main.py -m <model_file> -w <weight_file> -i <img_folder>'
			sys.exit()
		elif opt == "-m":
			model_filename = arg
		elif opt == "-w":
			weight_filename = arg
		elif opt == "-i":
			img_folder_name = arg

	print 'model file is "', model_filename
	print 'weight file is "', weight_filename
	print 'image file is "', img_folder_name


	#files/folders creation
	if os.path.isfile(model_filename):
	    print 'CaffeNet found.'
	else:
	    print 'Caffe model NOT found...'
	    sys.exit(2)


	img_mean_foldername = "mean_data"

	if not os.path.exists(img_mean_foldername):
		os.makedirs(img_mean_foldername)

	img_mean_filename = img_mean_foldername + '/mean'
		

	#CNN creation
	caffe.set_mode_cpu()

	net = caffe.Net(model_filename,      # defines the structure of the model
	               weight_filename,  # contains the trained weights
	               caffe.TEST)     # use test mode (e.g., don't perform dropout)


	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1)) #move image channels to outermost dimension 

	input_image_set = []

	for img_name in os.listdir(img_folder_name):

		image = caffe.io.load_image(img_folder_name + img_name)

		transformed_image = transformer.preprocess('data', image)

		input_image_set.append(transformed_image)



#	compute_mean(input_image_set, img_mean_filename)
#	mu = np.load(img_mean_filename + ".npy")
#	meanSubtractor = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
#	meanSubtractor.set_mean('data', mu)            # subtract the dataset-mean value in each channel
#	for img in input_image_set:
#		img = meanSubtractor.preprocess('data', img)


	#bs = len(input_image_set)  # batch size
	#in_shape = net.blobs['data'].data.shape
	#in_shape = [bs, in_shape[1], in_shape[2], in_shape[3]] # set new batch size
	#net.blobs['data'].reshape(*in_shape)
	#net.reshape()


	for layer_name, blob in net.blobs.iteritems():
		print layer_name + '\t' + str(blob.data.shape)



	for img in input_image_set:

		net.blobs['data'].data[...] = img

		out = net.forward()

		feat = net.blobs['fc25'].data[0]

		print feat

	#vis_square(feat)




if __name__=='__main__':	
	main(sys.argv[1:])