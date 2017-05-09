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


	caffe_root = '/home/dallo/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
	sys.path.insert(0, caffe_root + 'python')


	if os.path.isfile(model_filename):
	    print 'CaffeNet found.'
	else:
	    print 'Caffe model NOT found...'
	    sys.exit(2)



	caffe.set_mode_cpu()

	net = caffe.Net(model_filename,      # defines the structure of the model
	               weight_filename,  # contains the trained weights
	               caffe.TEST)     # use test mode (e.g., don't perform dropout)


	img_mean_filename = 'image_set_mean'
	#os.system("/home/dallo/CNN_SVM/compute_image_mean.py " +  img_mean_filename + " " + img_folder_name)
		



	for img_name in os.listdir(img_folder_name):


		transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
		#transformer.set_transpose('data', (2,0,1))

		net.blobs['data'].reshape(50,        # batch size
                          3,         # 3-channel (BGR) images
                          227, 227)  # image size is 227x227


		image = caffe.io.load_image(img_folder_name + img_name)



		transformed_image = transformer.preprocess('data', image)

		print img_name 
		print image.shape 
		print transformed_image.shape

if __name__=='__main__':	
	main(sys.argv[1:])