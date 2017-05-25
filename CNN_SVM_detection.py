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

from divide_et_impera import ImageCropper

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV


netLayers = {
    'googlenet': 'pool5/7x7_s1',
    #'googlenet': 'loss3/classifier',
    'vggnet':'fc8',
    #'vggnet': 'fc7',
    'resnet': 'fc1000',
    #'resnet': 'pool5'
}

interesting_labels = {'voc': ['aeroplane', 'bird', 'cat', 'boat', 'horse'],
                      'coco': ['snowboard', 'giraffe', 'cow', 'scissors', 'frisbee']}


def trainSVMsFromCroppedImages(net, networkName, train_imagesFolder, train_annotationsFolder, classes,  gridsearch = False):

    [filesTrainNames, imagesTrain, labelsTrain] = createSamplesDatastructures(train_imagesFolder, train_annotationsFolder, classes)

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

        print 'Opening old features from ', trainFeaturesFileName
        #Load features from a previously dumped file
        with open(trainFeaturesFileName, 'rb') as trainFeaturesFile:
            (filesTrainNames, featureVectorsTrain) = pickle.load(trainFeaturesFile)
            featureVectorsTrain = np.array(featureVectorsTrain)



    interestingIndices = [idx for idx, x in enumerate(labelsTrain) if x is not  'unknown']

    interestingLabelsTrain = [x for idx, x in enumerate(labelsTrain) if idx in interestingIndices]
    interestingFeaturesVectorTrain = [x for idx, x in enumerate(featureVectorsTrain) if idx in interestingIndices]

    inlier_outlierLabels = [1 if idx in interestingIndices else -1 for idx in range(len(labelsTrain))]


    featureVectorsTrain = normalizeData(featureVectorsTrain)

    interestingFeaturesVectorTrain = normalizeData(interestingFeaturesVectorTrain)

    print len(interestingFeaturesVectorTrain)



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
            clf = GridSearchCV(estimator, params, n_jobs=-1, cv=5, scoring="accuracy")
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

        #noveltyCLS = svm.OneClassSVM(nu=0.013, kernel = "rbf", gamma = 0.0078)
        #noveltyCLS.fit(interestingFeaturesVectorTrain)

        noveltyCLS = svm.SVC(C=621.017, kernel = "rbf")
        print len(featureVectorsTrain)
        print len(inlier_outlierLabels)
        print len(interestingFeaturesVectorTrain)
        print len(interestingLabelsTrain)
        noveltyCLS.fit(featureVectorsTrain, inlier_outlierLabels)



    multiclassSVM = svm.SVC(kernel="rbf", C=1e6, probability=True)
    multiclassSVM.fit(interestingFeaturesVectorTrain, interestingLabelsTrain)



    return [noveltyCLS, multiclassSVM]



def test(net, networkName, noveltyCLS, multiclassSVM, test_imagesFolder, test_annotationsFolder, classes):


    [filesTestNames, imagesTest, labelsTest] = createSamplesDatastructures(test_imagesFolder, test_annotationsFolder, classes)

    testFeaturesFileName = 'testFeatures' + networkName + '.b'

    if not os.path.isfile(testFeaturesFileName):
        imagesScale = 255.0

        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1)) #move image channels to outermost dimension 
        transformer.set_raw_scale('data', imagesScale)

        #Update the sets of images by transforming them according to Transformer
        for index in range(len(imagesTest)):
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

    truePredicted = []
    inlierPredicted = []

    for idx, isInlier in enumerate(isInliers):

        isInlier = int(isInlier)
        if isInlier == -1 and labelsTest[idx] == 'unknown':
            correctOutlier+=1
        if isInlier == 1 and labelsTest[idx] is not 'unknown':
            correctInlier+=1
            truePredicted.append(labelsTest[idx])
            inlierPredicted.append(predictions[idx])


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

    cnf_matrix = confusion_matrix(truePredicted, inlierPredicted)
    plot_confusion_matrix(cnf_matrix, classes = classes)

    print 'Accuracy: ', accuracy, ' Precision: ', precision, ' Recall: ', recall, ' Novelty precision: ', noveltyPrecision


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


    plt.tight_layout()
    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    plt.savefig("confusion_matrix.png")

    return




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

    trainPercentage = 0.7

    classes = interesting_labels[mode]

    print classes

    imageCropper = ImageCropper(images_dir, annotations_dir,train_imagesFolder, train_annotationsFolder,test_imagesFolder,test_annotationsFolder, mode)

   # if mode == 'coco':
   #    imageCropper.downloadCoco(classes,500)



    trainList, testList = imageCropper.splitTrainTest(classes, trainPercentage)


    imageCropper.extractBBoxesImages(trainList, 'train')
    imageCropper.extractBBoxesImages(testList, 'test')

    if mode == 'coco':
        classes = imageCropper.getCocoCategoriesId(classes)

    noveltyCLS, multiclassSVM = trainSVMsFromCroppedImages(net, cnn_type, train_imagesFolder,
                                                           train_annotationsFolder, classes, gridsearch)

    test(net, cnn_type, noveltyCLS, multiclassSVM, test_imagesFolder, test_annotationsFolder, classes)

    

if __name__=='__main__':
    main(sys.argv[1:])
