# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import os
os.environ['GLOG_minloglevel'] = '3' 

import caffe
import itertools
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
from sklearn import svm
import time
import argparse
from sklearn.metrics import confusion_matrix
from dataset_utils import createSamplesDatastructures, normalizeData, readLabelFromAnnotation

from divide_et_impera import extractBBoxesImages, splitTrainTest

from caffe.io import array_to_blobproto
from collections import defaultdict
from skimage import io
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV


netLayers = {
    'googlenet':'pool5/7x7_s1',
    'vggnet':'fc8',
    'resnet':'fc1000'

}

interesting_labels = {'voc':['aeroplane','bird','cat','boat','horse']}



def trainSVMsFromCroppedImages(net, networkName, trainList, images_dir_in, annotations_dir_in, images_dir_out, annotations_dir_out,mode, gridsearch = False):

    classes = interesting_labels[mode]

    extractBBoxesImages(trainList,images_dir_in,annotations_dir_in, images_dir_out, annotations_dir_out)

    [filesTrainNames, imagesTrain, labelsTrain] = createSamplesDatastructures(images_dir_out, annotations_dir_out, classes, mode)

    trainFeaturesFileName = 'trainFeatures' + networkName + '.b'

    if not os.path.isfile(trainFeaturesFileName):

        imagesScale = 255.0

        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1)) #move image channels to outermost dimension 
        transformer.set_raw_scale('data', imagesScale) 

        #Update the sets of images by transforming them according to Transformer
        for  index in range(len(imagesTrain)):
            imagesTrain[index] = transformer.preprocess('data', imagesTrain[index])

        extractionLayerName = netLayers[networkName]
        t1 = time.time()
        featureVectorsTrain = extractFeatures(imagesTrain, net, extractionLayerName)
        print '\nFeatures extraction took ',(time.time() - t1) ,' seconds for ', len(imagesTrain), ' images'

        #Dump features in a file 
        with open(trainFeaturesFileName, 'wb') as trainFeaturesFile:
            pickle.dump((filesTrainNames, featureVectorsTrain), trainFeaturesFile)



    else:

        print 'Opening old features.... '
        #Load features from a previously dumped file
        with open(trainFeaturesFileName, 'rb') as trainFeaturesFile:
            (filesTrainNames, featureVectorsTrain) = pickle.load(trainFeaturesFile)
            featureVectorsTrain = np.array(featureVectorsTrain)



	interestingIndices = [idx for idx, x in enumerate(labelsTrain) if x is not  'unknown']

	interestingImagesTrain = [x for idx, x in enumerate(imagesTrain) if idx in interestingIndices]
	interestingLabelsTrain = [x for idx, x in enumerate(labelsTrain) if idx in interestingIndices]
	interestingFeaturesVectorTrain = [x for idx, x in enumerate(featureVectorsTrain) if idx in interestingIndices]

	inlier_outlierLabels = [1 if idx in interestingIndices else -1 for idx in range(len(labelsTrain))]


	featureVectorsTrain = normalizeData(featureVectorsTrain)

	interestingFeaturesVectorTrain = normalizeData(interestingFeaturesVectorTrain)


    if gridsearch:

		nu =  [x for x in np.logspace(-4, 0, 20)]  
		gamma = [x for x in np.logspace(-4,0,20)]
		C = [x for x in np.logspace(-1, 10, 30)]
		n_estimators = [int(round(x)) for x in np.logspace(1, 5, 20)]
		contamination = [x for x in np.linspace(0, 0.5, 10)]
		classifiers = {
		"oneClass": (svm.OneClassSVM(),{"nu": nu,
		"gamma": gamma}),
		"2Class": (svm.SVC(),{"C": C}),
		"Forest": (IsolationForest(),{"n_estimators": n_estimators,
		"contamination": contamination}	)	}

		score = 0

	
		for name_estimator, (estimator, params) in classifiers.iteritems():
		    print name_estimator
		    clf = GridSearchCV(estimator, params, n_jobs = -1, cv = 5, scoring = "accuracy")
		    if name_estimator is "oneClass" or name_estimator is "Forest":
		        
		        trainDataSet = np.asarray(interestingFeaturesVectorTrain)
		        labels = [1 for x in range(len(interestingFeaturesVectorTrain))]
		      
		        clf.fit(trainDataSet, labels)
		   
		    else:
		        
		        trainDataSet = featureVectorsTrain
		        labels = inlier_outlierLabels

		        clf.fit(trainDataSet, labels)

		    print "Estimator: ", name_estimator, "\n", clf.best_params_, " score: ", clf.best_score_   

		    if clf.best_score_ > score:
		        score = clf.best_score_
		        noveltyCLS = clf.best_estimator_


    else:

		noveltyCLS = svm.OneClassSVM(nu=0.013, kernel = "rbf", gamma = 0.0078)
		noveltyCLS.fit(interestingFeaturesVectorTrain)
        
		#noveltyCLS = svm.SVC(C=621.017, kernel = "rbf")
		#noveltyCLS.fit(featureVectorsTrain, inlier_outlierLabels)
    


    multiclassSVM = svm.SVC(kernel="rbf", C=1e6, probability=True)
    multiclassSVM.fit(interestingFeaturesVectorTrain, interestingLabelsTrain)



    return [noveltyCLS, multiclassSVM]



def test(net, networkName, noveltyCLS, multiclassSVM, testList, images_dir_in, annotations_dir_in, images_dir_out, annotations_dir_out,mode):

    classes = interesting_labels[mode]

    extractBBoxesImages(testList,images_dir_in,annotations_dir_in, images_dir_out, annotations_dir_out)

    [filesTestNames, imagesTest, labelsTest] = createSamplesDatastructures(images_dir_out, annotations_dir_out, classes, mode)

    testFeaturesFileName = 'testFeatures' + networkName + '.b'

    if not os.path.isfile(testFeaturesFileName):
        imagesScale = 255.0

        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1)) #move image channels to outermost dimension 
        transformer.set_raw_scale('data', imagesScale) 

        #Update the sets of images by transforming them according to Transformer
        for  index in range(len(imagesTest)):
            imagesTest[index] = transformer.preprocess('data', imagesTest[index])

        
        extractionLayerName = netLayers[networkName]
        t1 = time.time()
        featureVectorsTest = extractFeatures(imagesTest, net, extractionLayerName)
        print '\nFeatures extraction took ',(time.time() - t1) ,' seconds for ', len(imagesTest), ' images'

        #Dump features in a file 
        with open(testFeaturesFileName, 'wb') as testFeaturesFile:
            pickle.dump((filesTestNames, featureVectorsTest), testFeaturesFile)



    else:

        print 'Opening old features.... '
        #Load features from a previously dumped file
        with open(testFeaturesFileName, 'rb') as testFeaturesFile:
            (filesTestNames, featureVectorsTest) = pickle.load(testFeaturesFile)
            featureVectorsTest = np.array(featureVectorsTest)


    featureVectorsTest = normalizeData(featureVectorsTest)

    correctOutlier = 0
    correctInlier = 0
    correctClass = 0
    numPredicted = 0

    isInliers = noveltyCLS.predict(featureVectorsTest)
    predictions = multiclassSVM.predict(featureVectorsTest)
	
	
    for idx, isInlier in enumerate(isInliers):
        
        isInlier = int(isInlier)		
        if isInlier == -1 and labelsTest[idx] == 'unknown':
            correctOutlier+=1
        if isInlier == 1 and labelsTest[idx] is not 'unknown':
            correctInlier+=1
        

        if isInlier == 1:
            numPredicted+=1
            if predictions[idx] == labelsTest[idx]:
                correctClass+=1

    numInterestingSamples = sum(i is not 'unknown' for i in labelsTest)
    numSamples = len(labelsTest)    
    print 'num interesting labels {}\nunknown {}\ntotal {}\ncorrect outliers {}\ncorrect inlier {}'.format(numInterestingSamples, numSamples-numInterestingSamples, numSamples, correctOutlier, correctInlier)
    
    precision = 100.*correctClass/numPredicted
    recall = 100.*correctClass/numInterestingSamples
    accuracy = 100.*(correctClass + correctOutlier)/numSamples
    noveltyPrecision = 100.* (correctOutlier + correctInlier)/numSamples

    print 'Accuracy: ', accuracy, ' Precision: ', precision, ' Recall: ', recall


def backspace(n):
    sys.stdout.write('\r'+n)
    sys.stdout.flush()


def extractFeatures(imageSet, net, extractionLayerName):

    featuresVector = []
    totalImages = len(imageSet)
    for num, image in enumerate(imageSet):

        net.blobs['data'].data[...] = image
        net.forward()
        features = net.blobs[extractionLayerName].data[0]
        featuresVector.append(features.copy().flatten())
        string_to_print = '{} of {}'.format(num, totalImages)
        backspace(string_to_print)

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
    cnn_type = ''
    mode = 'voc'
    images_dir = 'VOC2007/JPEGImages'
    annotations_dir = 'VOC2007/Annotations'
    gridsearch = False
    caffe.set_mode_cpu()

    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights", help="caffemodel file")
    parser.add_argument("-p", "--prototxt", help="prototxt file")
    parser.add_argument("-i", "--input_im", help="input images dir")
    parser.add_argument("-a", "--annotations_dir", help="input annotations dir")
    parser.add_argument("-t", "--dataset_type", help="dataset type (voc/imagenet/coco)")    
    parser.add_argument("-n", "--net_type", help="cnn type (resnet/googlenet/vggnet")
    parser.add_argument("-g", "--gpu", help="enable gpu mode", action='store_true')
    parser.add_argument("-s", "--search", help="enable gridsearch", action='store_true')
    args = parser.parse_args()

    if args.prototxt:
        model_filename = args.prototxt
    if args.weights:
        weight_filename = args.weights
    if args.input_im:
        images_dir = args.input_im
    if args.annotations_dir:
        annotations_dir = args.annotations_dir
    if args.dataset_type:
    	mode = args.dataset_type
    if args.net_type:
    	cnn_type = args.net_type
    if args.gpu:
		caffe.set_mode_gpu()
		caffe.set_device(0)
    if args.search:
        gridsearch = True


    print 'model file is ', model_filename
    print 'weight file is ', weight_filename
    print 'images dir is ', images_dir
    print 'annotations dir is ', annotations_dir
    print 'the cnn is ', cnn_type	    

    if os.path.isfile(model_filename):
        print 'Caffe model found.'
    else:
        print 'Caffe model NOT found...'
        sys.exit(2)

    net = caffe.Net(model_filename,      # defines the structure of the model
                   weight_filename,  # contains the trained weights
                  caffe.TEST)     # use test mode (e.g., don't perform dropout)


    train_imagesFolder = 'train_images'
    train_annotationsFolder = 'train_annotations'

    test_imagesFolder = 'test_images'
    test_annotationsFolder = 'test_annotations'

    train_testPercentage = 0.7

    [trainList, testList] = splitTrainTest(annotations_dir, classes, train_testPercentage)

    [noveltyCLS, multiclassSVM] = trainSVMsFromCroppedImages(net, cnn_type, trainList, images_dir,annotations_dir,  train_imagesFolder, train_annotationsFolder, mode, gridsearch)
    
    test(net, cnn_type, noveltyCLS, multiclassSVM, testList,images_dir,annotations_dir, test_imagesFolder,  test_annotationsFolder, mode)



if __name__=='__main__':	
    main(sys.argv[1:])
