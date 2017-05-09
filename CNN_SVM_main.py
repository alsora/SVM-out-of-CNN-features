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


	#CNN data layer specification
	net.blobs['data'].reshape(50,        # batch size
                  3,         # 3-channel (BGR) images
                  227, 227)  # image size is 227x227


	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_raw_scale('data', 255) #set pixel values in range [0,255]
	transformer.set_transpose('data', (2,0,1)) #Set BGR 

	input_image_set = []

	for img_name in os.listdir(img_folder_name):

		image = caffe.io.load_image(img_folder_name + img_name)

		transformed_image = transformer.preprocess('data', image)

		input_image_set.append(transformed_image)



	compute_mean(input_image_set, img_mean_filename)


	mu = np.load(img_mean_filename + ".npy")


	meanSubtractor = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

	meanSubtractor.set_mean('data', mu)            # subtract the dataset-mean value in each channel

	#for img in input_image_set:

		#img = meanSubtractor.preprocess('data', img)


	#out = net.forward_all(data=np.asarray([transformer.preprocess('data', input_image_set)]))






if __name__=='__main__':	
	main(sys.argv[1:])