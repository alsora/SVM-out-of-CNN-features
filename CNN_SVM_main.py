# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import caffe
# GPU_ID = 0 # Switch between 0 and 1 depending on the GPU you want to use.
# caffe.set_mode_gpu()
# caffe.set_device(GPU_ID)
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

from caffe.io import array_to_blobproto
from collections import defaultdict
from skimage import io

from compute_mean import compute_mean 



def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    print data.shape
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
        
    print data.shape

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    print data.shape

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    print data.shape

    plt.imshow(data); plt.axis('off')

	#cv2.waitKey(0)



def splitDataset(imgSetName, trainPercentage):


	#Open allNamesListFile, and split the data into:

	imagesTrainSet = "ImagesTrainSet.txt"  
	imagesTestSet = "ImagesTestSet.txt"
	imgSet = []
	trainSet = []
	testSet = []

	numElements = 0

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

	trainFile = open(imagesTrainSet, "w")
	trainFile.write("\n".join(trainSet))
	trainFile.close()

	testFile = open(imagesTestSet, "w")
	testFile.write("\n".join(testSet))
	testFile.close()

    

	return [imagesTrainSet, imagesTestSet]



def createSamplesDatastructures(samplesListFileName, interesting_labels):

	imagesFolderPath = "VOC2007/JPEGImages/"
	annotationsFolderPath = "VOC2007/Annotations/"

	samplesNames = []
	samplesImages = []
	samplesLabels = []

	with open(samplesListFileName,'r') as fTrain:
		for sampleName in (fTrain.read().splitlines()): 
			
			samplesNames.append(sampleName)

			imageCompletePath = imagesFolderPath + sampleName + '.jpg'
			image = caffe.io.load_image(imageCompletePath)
			samplesImages.append(image)

			annotationCompletePath = annotationsFolderPath + sampleName + '.xml'
			label = readLabelFromAnnotation(annotationCompletePath, interesting_labels)
			samplesLabels.append(label)

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
	try:
		opts, args = getopt.getopt(argv, "hm:w:i:")
		print opts
	except getopt.GetoptError:
		print 'CNN_SVM_main.py -m <model_file> -w <output_file> -i <img_files_list>'
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print 'CNN_SVM_main.py -m <model_file> -w <weight_file> -i <img_files_list>'
			sys.exit()
		elif opt == "-m":
			model_filename = arg
		elif opt == "-w":
			weight_filename = arg
		elif opt == "-i":
			listImagesNames = arg

	print 'model file is "', model_filename
	print 'weight file is "', weight_filename
	print 'image file is "', listImagesNames


	caffe.set_mode_cpu()

	if os.path.isfile(model_filename):
	    print 'Caffe model found.'
	else:
	    print 'Caffe model NOT found...'
	    sys.exit(2)

	trainDataPercentage = 0.7
	
	interesting_labels = ['aeroplane','bird','cat','boat','horse']
	trainFeaturesFileName = 'trainFeatures.b'
	testFeaturesFileName =  'testFeatures.b'


	#Given file listImagesNames and percentage-> listImagesNamesTrain listImagesNamesTest
	[listImagesNamesTrain, listImagesNamesTest] = splitDataset(listImagesNames, trainDataPercentage)



	#Load all images in data structure -> imagesTrain imagesTest
	[filesTrainNames, imagesTrain, labelsTrain] = createSamplesDatastructures(listImagesNamesTrain, interesting_labels)
	[filesTestNames, imagesTest, labelsTest] = createSamplesDatastructures(listImagesNamesTest, interesting_labels)


	#CNN creation
	net = caffe.Net(model_filename,      # defines the structure of the model
	               weight_filename,  # contains the trained weights
	              caffe.TEST)     # use test mode (e.g., don't perform dropout)


	extractionLayerName = 'fc25'
	if extractionLayerName not in net.blobs:
		raise TypeError("Invalid layer name: " + extractionLayerName)

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


	'''
	filters = net.params['conv1'][0].data
	vis_square(filters.transpose(0, 2, 3, 1))

	vis_square(featureVectorsTest[0])
	'''

if __name__=='__main__':	
	main(sys.argv[1:])