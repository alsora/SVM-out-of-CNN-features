# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import caffe

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import sys, getopt
import cv2
import os
import pickle
from sklearn import svm
import time
import random
import xml.etree.ElementTree as ET
import hashlib

from caffe.io import array_to_blobproto
from collections import defaultdict
from skimage import io

from compute_mean import compute_mean 





def vis_square(data, filename=''):
	"""Take an array of shape (n, height, width) or (n, height, width, 3)
	and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
	
	print 'The data you are going to visualize has shape ', data.shape
	
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

	plt.axis('off')
	plt.imshow(data)

	# Save plot if filename is set
	if filename != '':
		plt.savefig(filename, bbox_inches='tight', pad_inches=0)

	plt.show()


def splitDataset(imgSetName, trainPercentage, partialMd5CheckSum):


	#Open allNamesListFile, and split the data into:

	imagesTrainSetName = 'ImagesTrainSet_' + partialMd5CheckSum + '.txt'  
	imagesTestSetName = 'ImagesTestSet_' + partialMd5CheckSum + '.txt'  
	imgSet = []
	trainSet = []
	testSet = []

	numElements = 0

	if not os.path.isfile(imagesTrainSetName) or not os.path.isfile(imagesTestSetName):

		with open(imgSetName, 'r') as file:
			for sampleName in (file.read().splitlines()):
				imgSet.append(sampleName)
				numElements+=1

		rangeImageIndices = range(numElements)
		numberTrainSample = int(round(trainPercentage*numElements))   
		trainIndices = random.sample(rangeImageIndices, numberTrainSample) 
		countTrain = 0
		countTest = 0

		for i in range(numElements):
			if i in trainIndices:
				trainSet.append(imgSet[i])
				countTrain += 1
			else:
				testSet.append(imgSet[i])
				countTest += 1

		random.shuffle(trainSet)
		random.shuffle(testSet)

		trainFile = open(imagesTrainSetName, "w")
		trainFile.write("\n".join(trainSet))
		trainFile.close()

		testFile = open(imagesTestSetName, "w")
		testFile.write("\n".join(testSet))
		testFile.close()

    

	return [imagesTrainSetName, imagesTestSetName]



def createSamplesDatastructures(samplesListFileName, interesting_labels, mode):


	samplesNames = []
	samplesImages = []
	samplesLabels = []


	if mode == 'voc':

		imagesFolderPath = "VOC2007/JPEGImages/"
		annotationsFolderPath = "VOC2007/Annotations/"

		with open(samplesListFileName,'r') as file:
			for sampleName in (file.read().splitlines()): 
				
				samplesNames.append(sampleName)

				imageCompletePath = imagesFolderPath + sampleName + '.jpg'
				image = caffe.io.load_image(imageCompletePath)
				samplesImages.append(image)

				annotationCompletePath = annotationsFolderPath + sampleName + '.xml'
				label = readLabelFromAnnotation(annotationCompletePath, interesting_labels)
				samplesLabels.append(label)

		return [samplesNames, samplesImages, samplesLabels]

	elif mode == 'imagenet':

		with open(samplesListFileName,'r') as file:
			for samplePath in (file.read().splitlines()): 
				
				splittedName = sampleName.split('/')

				samplesNames.append(splittedName[1])

				image = caffe.io.load_image(samplePath)
				samplesImages.append(image)

				samplesLabels.append(splittedName[0])

		return [samplesNames, samplesImages, samplesLabels]




def readLabelFromAnnotation(annotationFileName, interesting_labels):
	#Parse the given annotation file and read the label

	label = []
	tree = ET.parse(annotationFileName)
	root = tree.getroot()
	for obj in root.findall('object'):
		name = obj.find('name').text

		if (name in interesting_labels):
			label.append(name)
			break


	return label



def extractFeatures(imageSet, net, extractionLayerName):

	featuresVector = []
	
	for image in imageSet:
		net.blobs['data'].data[...] = image
		net.forward()
		features = net.blobs[extractionLayerName].data[0]
		featuresVector.append(features.copy().flatten())

	return featuresVector




def main(argv):
	
	model_filename = ''
	weight_filename = ''
	img_filename = ''
	mode = ' '
	try:
		opts, args = getopt.getopt(argv, "hm:w:i:s:")
		print opts
	except getopt.GetoptError:
		print 'CNN_SVM_main.py -m <model_file> -w <output_file> -i <img_files_list> -s <mode>'
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print 'CNN_SVM_main.py -m <model_file> -w <weight_file> -i <img_files_list> -s <mode>'
			sys.exit()
		elif opt == "-m":
			model_filename = arg
		elif opt == "-w":
			weight_filename = arg
		elif opt == "-i":
			listImagesNames = arg
		elif opt == "-s":
			mode = arg;

	print 'model file is ', model_filename
	print 'weight file is ', weight_filename
	print 'image file is ', listImagesNames
	print 'mode is ', mode



	interesting_labels = ['aeroplane','bird','cat','boat','horse']
	trainDataPercentage = 0.7
	

	if os.path.isfile(model_filename):
	    print 'Caffe model found.'
	else:
	    print 'Caffe model NOT found...'
	    sys.exit(2)


	if not mode == 'voc' and not mode == 'imagenet':
		print 'The given mode is not supported...'
		sys.exit(2)


	caffe.set_mode_cpu()

	#CNN creation
	net = caffe.Net(model_filename,      # defines the structure of the model
	               weight_filename,  # contains the trained weights
	              caffe.TEST)     # use test mode (e.g., don't perform dropout)

	extractionLayerName = 'fc25'
	if extractionLayerName not in net.blobs:
		raise TypeError("Invalid layer name: " + extractionLayerName)
	



	md5CheckSum = hashlib.md5(open(listImagesNames,'rb').read()).hexdigest()
	partialMd5CheckSum = md5CheckSum[:5]
	

	#Given file listImagesNames and percentage-> listImagesNamesTrain listImagesNamesTest
	[listImagesNamesTrain, listImagesNamesTest] = splitDataset(listImagesNames, trainDataPercentage, partialMd5CheckSum)


	#Load all images in data structure -> imagesTrain imagesTest
	[filesTrainNames, imagesTrain, labelsTrain] = createSamplesDatastructures(listImagesNamesTrain, interesting_labels, mode)
	[filesTestNames, imagesTest, labelsTest] = createSamplesDatastructures(listImagesNamesTest, interesting_labels, mode)


	trainFeaturesFileName = 'trainFeatures_' + partialMd5CheckSum + '.b'
	testFeaturesFileName =  'testFeatures_' + partialMd5CheckSum + '.b'

	if not os.path.isfile(trainFeaturesFileName) or not os.path.isfile(testFeaturesFileName):


		transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
		transformer.set_transpose('data', (2,0,1)) #move image channels to outermost dimension 

		#Update the sets of images by transforming them according to Transformer
		for  index in range(len(imagesTrain)):
			imagesTrain[index] = transformer.preprocess('data', imagesTrain[index])

		for  index in range(len(imagesTest)):
			imagesTest[index] = transformer.preprocess('data', imagesTest[index])		


		#Forward the net on all images -> featureVectorsTrain featureVectorsTest
		featureVectorsTrain = []
		featureVectorsTest = []
		
		t1 = time.time()
		featureVectorsTrain = extractFeatures(imagesTrain, net, extractionLayerName)
		featureVectorsTest = extractFeatures(imagesTest, net, extractionLayerName)
		print 'Features extraction took ',(time.time() - t1) ,' seconds for ', (len(imagesTrain) + len(imagesTest)), ' images'

		#Dump features in 2 files -> trainFeatures testFeatures
		with open(trainFeaturesFileName, 'wb') as trainFeaturesFile:
			pickle.dump((filesTrainNames, featureVectorsTrain), trainFeaturesFile)

		with open(testFeaturesFileName, 'wb') as testFeaturesFile:
			pickle.dump((filesTestNames, featureVectorsTest), testFeaturesFile)	



	else:

		#Load features from a previously dumped file
		with open(trainFeaturesFileName, 'rb') as trainFeaturesFile:
			(filesTrainNames, featureVectorsTrain) = pickle.load(trainFeaturesFile)
			featureVectorsTrain = np.array(featureVectorsTrain)

		with open(testFeaturesFileName, 'rb') as testFeaturesFile:
			(filesTestNames, featureVectorsTest) = pickle.load(testFeaturesFile)
			featureVectorsTest = np.array(featureVectorsTest)






	#Fit a SVM model on the extracted trainFeatures
	t1 = time.time()
	modelSVM = svm.SVC(kernel="linear", C=1e6, probability=True) # little regularization needed - get this training set right, neglect margin
	modelSVM.fit(featureVectorsTrain, labelsTrain)
	print 'SVM training took ',(time.time() - t1) ,' seconds'



	#Test the SVM using the extracted testFeatures
	t1 = time.time()
	for index in range(len(filesTestNames)):
		features =  np.array(featureVectorsTest[index]).reshape((1, -1))
		prediction = modelSVM.predict(features)
		print 'For image ', filesTestNames[index], ' ... Predicted: ', prediction, ' TrueLabel: ', labelsTest[index]

	print 'SVM test took ',(time.time() - t1) ,' seconds'




	filters = net.params['conv1'][0].data
	vis_square(filters.transpose(0, 2, 3, 1), filename='conv1.png')


	#activatedFeatures = net.blobs['conv1'].data #Not working on activation features....
	#vis_square(activatedFeatures.transpose(0, 2, 3, 1))
	

if __name__=='__main__':	
	main(sys.argv[1:])