# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import os
os.environ['GLOG_minloglevel'] = '3' 

import caffe
import itertools
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import sys, getopt
import cv2
import pickle
from sklearn import svm
import time
import random
import xml.etree.ElementTree as ET
import hashlib
import glob
from sklearn.metrics import confusion_matrix

from divide_et_impera import extractBBoxesImages

from caffe.io import array_to_blobproto
from collections import defaultdict
from skimage import io

from compute_mean import compute_mean 







def createSamplesDatastructures(images_dir, annotations_dir, mode):


    samplesNames = []
    samplesImages = []
    samplesLabels = []


    if mode == 'voc':

        for root, dirs, files in os.walk(images_dir):
            for image_name in files:
                name, extension = image_name.split(".")

                samplesNames.append(name)

                imageCompletePath = images_dir + '/' + image_name
                image = caffe.io.load_image(imageCompletePath)
                samplesImages.append(image)

                annotationCompletePath = annotations_dir + '/' + name + '.xml'
                label = readLabelFromAnnotation(annotationCompletePath)
                samplesLabels.append(label)

        imagesFolderPath = images_dir
        annotationsFolderPath = annotations_dir


        return [samplesNames, samplesImages, samplesLabels]




def trainSVMsFromCroppedImages(net, extractionLayerName, images_dir, annotations_dir,interesting_labels):

    extractBBoxesImages("VOC2007/Annotations", annotations_dir, "VOC2007/JPEGImages", images_dir, interesting_labels)

    [filesTrainNames, imagesTrain, labelsTrain] = createSamplesDatastructures(images_dir, annotations_dir, 'voc')

    trainFeaturesFileName = 'trainFeatures.b'

    if not os.path.isfile(trainFeaturesFileName):

        imagesScale = 255.0

        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1)) #move image channels to outermost dimension 
        transformer.set_raw_scale('data', imagesScale) 

        #Update the sets of images by transforming them according to Transformer
        for  index in range(len(imagesTrain)):
            imagesTrain[index] = transformer.preprocess('data', imagesTrain[index])

        featureVectorsTrain = []

        t1 = time.time()
        featureVectorsTrain = extractFeatures(imagesTrain, net, extractionLayerName)
        print 'Features extraction took ',(time.time() - t1) ,' seconds for ', len(imagesTrain), ' images'

        #Dump features in a file 
        with open(trainFeaturesFileName, 'wb') as trainFeaturesFile:
            pickle.dump((filesTrainNames, featureVectorsTrain), trainFeaturesFile)

    else:

        print 'Opening old features.... '
        #Load features from a previously dumped file
        with open(trainFeaturesFileName, 'rb') as trainFeaturesFile:
            (filesTrainNames, featureVectorsTrain) = pickle.load(trainFeaturesFile)
            featureVectorsTrain = np.array(featureVectorsTrain)


    featureVectorsTrainNormalized = []

    for vec in featureVectorsTrain:
        vecNormalized = vec/np.linalg.norm(vec)
        featureVectorsTrainNormalized.append(vecNormalized)

    trainMean = np.mean(featureVectorsTrainNormalized, axis = 0)

    featureVectorsTrainNormalizedCentered = []

    for vec in featureVectorsTrainNormalized:
        vecCentered = vec - trainMean
        featureVectorsTrainNormalizedCentered.append(vecCentered)


    #Creates and train 2 svms: a novelty detector and a multi class classifier one
    noveltySVM = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    multiclassSVM = svm.SVC(kernel="rbf", C=1e6, probability=True) # little regularization needed - get this training set right, neglect margin

    t1 = time.time()
    noveltySVM.fit(featureVectorsTrainNormalizedCentered)
    multiclassSVM.fit(featureVectorsTrainNormalizedCentered, labelsTrain)
    print 'SVMs training took ',(time.time() - t1) ,' seconds'



    return [noveltySVM, multiclassSVM]



def testImages():

    #create test set with annotated images
    #for each image crop it according to ALL the bounding boxes in the annotation (using divide_et_impera)
    #you will get a lot of cropped images, some belonging to interesting classes and some not.

    #load cropped images
    #assign a label to each image. If the image is not referring to an interesting label assign unknown
    #for each cropped image just created 
    featuresVector = extractFeatures(croppedImageSet, net, extractionLayerName)
    for index in len(featuresVector):

        isInlier = noveltySVM.predict(featuresVector[index])
        #predict should return +1 or -1

        if isInlier:
            prediction = multiclassSVM.predict(featuresVector[index])

    #accuracy precision recall
    #we need a score for each svm. noveltySVM check if -1 corresponds to unknown
    #multiclassSVM check if class correspond to label        





def readLabelFromAnnotation(annotationFileName):
    #Parse the given annotation file and read the label

    tree = ET.parse(annotationFileName)
    root = tree.getroot()
    for obj in root.findall('object'):
        name = obj.find('name').text

        return name


def extractFeatures(imageSet, net, extractionLayerName):

    featuresVector = []

    for image in imageSet:
        net.blobs['data'].reshape(1,3,227,227)
        net.blobs['data'].data[...] = image
        net.forward()
        features = net.blobs[extractionLayerName].data[0]
        featuresVector.append(features.copy().flatten())

    return featuresVector


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    np.set_printoptions(precision=2)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, np.around(cm[i, j], decimals=2),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #print(cm)

    plt.tight_layout()
    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    plt.savefig("confusion_matrix.png")




def main(argv):

    model_filename = ''
    weight_filename = ''
    images_dir = ''
    annotations_dir = ''
    try:
        opts, args = getopt.getopt(argv, "hm:w:i:a:")
        print opts
    except getopt.GetoptError:
        print 'CNN_SVM_main.py -m <model_file> -w <weight_file> -i <images_dir> -a <annotations_dir>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'CNN_SVM_main.py -m <model_file> -w <weight_file> -i <images_dir> -s <annotations_dir>'
            sys.exit()
        elif opt == "-m":
            model_filename = arg
        elif opt == "-w":
            weight_filename = arg
        elif opt == "-i":
            images_dir = arg
        elif opt == "-a":
            annotations_dir = arg;

    print 'model file is ', model_filename
    print 'weight file is ', weight_filename
    print 'images dir is ', images_dir
    print 'annotations dir is ', annotations_dir



    interesting_labels = ['aeroplane','bird','cat','boat','horse']


    if os.path.isfile(model_filename):
        print 'Caffe model found.'
    else:
        print 'Caffe model NOT found...'
        sys.exit(2)


    caffe.set_mode_cpu()

    #CNN creation
    net = caffe.Net(model_filename,      # defines the structure of the model
                   weight_filename,  # contains the trained weights
                  caffe.TEST)     # use test mode (e.g., don't perform dropout)
    #os.environ['GLOG_minloglevel'] = '0' 

    extractionLayerName = 'fc25'
    if extractionLayerName not in net.blobs:
        raise TypeError("Network " +model_filename + " does not contain layer with name: " + extractionLayerName)


    [noveltySVM, multiclassSVM] = trainSVMsFromCroppedImages(net, extractionLayerName, images_dir, annotations_dir,interesting_labels)
   


if __name__=='__main__':	
    main(sys.argv[1:])
