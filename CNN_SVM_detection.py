def main(argv):

    model_filename = ''
    weight_filename = ''
    img_filename = ''
    mode = ' '
    try:
        opts, args = getopt.getopt(argv, "hm:w:i:s:")
        print opts
    except getopt.GetoptError:
        print 'CNN_SVM_main.py -m <model_file> -w <weight_file> -i <img_files_list> -s <mode>'
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
    #os.environ['GLOG_minloglevel'] = '0' 

    extractionLayerName = 'fc25'
    if extractionLayerName not in net.blobs:
        raise TypeError("Network " +model_filename + " does not contain layer with name: " + extractionLayerName)




    md5CheckSum = hashlib.md5(open(listImagesNames,'rb').read()).hexdigest()
    partialMd5CheckSum = md5CheckSum[:5]

    # Import the cropped images to be forwarded to the net


    #Given file listImagesNames and percentage-> listImagesNamesTrain listImagesNamesTest
    [listImagesNamesTrain, listImagesNamesTest] = splitDataset(listImagesNames, trainDataPercentage, partialMd5CheckSum)


    #Load all images in data structure -> imagesTrain imagesTest
    [filesTrainNames, imagesTrain, labelsTrain] = createSamplesDatastructures(listImagesNamesTrain, interesting_labels, mode)
    [filesTestNames, imagesTest, labelsTest] = createSamplesDatastructures(listImagesNamesTest, interesting_labels, mode)


    trainFeaturesFileName = 'trainFeatures_' + partialMd5CheckSum + '.b'
    testFeaturesFileName =  'testFeatures_' + partialMd5CheckSum + '.b'

    if not os.path.isfile(trainFeaturesFileName) or not os.path.isfile(testFeaturesFileName):

    	#YOLO CNN uses images in range [0,1], while other models (VGG, faster RCNN ...) use images in range [0,255]
    	if 'yolo' not in model_filename:
    		imagesScale = 1.0;
    	else:
    		imagesScale = 255.0

        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1)) #move image channels to outermost dimension 
        transformer.set_raw_scale('data', imagesScale) 

        if mode is 'imagenet' and 'yolo' not in model_filename:
        	imagenetMeanPath = caffe.__path__[0] + '/imagenet/ilsvrc_2012_mean.npy'
        	imagenetMean = np.load(imagenetMeanPath)
        	imagenetMean = imagenetMean/(255.0/imagesScale)
        	transformer.set_mean('data',imagenetMean.mean(1).mean(1))


        #Update the sets of images by transforming them according to Transformer
        for  index in range(len(imagesTrain)):
            imagesTrain[index] = transformer.preprocess('data', imagesTrain[index])

        for  index in range(len(imagesTest)):
            imagesTest[index] = transformer.preprocess('data', imagesTest[index])		


        #Forward the net on all images -> featureVectorsTrain featureVectorsTest
        featureVectorsTrain = []
        featureVectorsTest = []

        # Extract the full feature vector from last fc layer once cropped images are forwarded to the net

        t1 = time.time()
        featureVectorsTrain = extractFeatures(imagesTrain, net, extractionLayerName)
        featureVectorsTest = extractFeatures(imagesTest, net, extractionLayerName)
        print 'Features extraction took ',(time.time() - t1) ,' seconds for ', (len(imagesTrain) + len(imagesTest)), ' images'

        oldTrainFeaturesFile = glob.glob('trainFeatures_*')
        oldTestFeaturesFile = glob.glob('testFeatures_*')

        for file in oldTrainFeaturesFile:
            os.remove(file)

        for file in oldTestFeaturesFile:
            os.remove(file)


        #Dump features in 2 files -> trainFeatures testFeatures
        with open(trainFeaturesFileName, 'wb') as trainFeaturesFile:
            pickle.dump((filesTrainNames, featureVectorsTrain), trainFeaturesFile)

        with open(testFeaturesFileName, 'wb') as testFeaturesFile:
            pickle.dump((filesTestNames, featureVectorsTest), testFeaturesFile)	


    else:

        print 'Opening old features.... '

        #Load features from a previously dumped file
        with open(trainFeaturesFileName, 'rb') as trainFeaturesFile:
            (filesTrainNames, featureVectorsTrain) = pickle.load(trainFeaturesFile)
            featureVectorsTrain = np.array(featureVectorsTrain)

        with open(testFeaturesFileName, 'rb') as testFeaturesFile:
            (filesTestNames, featureVectorsTest) = pickle.load(testFeaturesFile)
            featureVectorsTest = np.array(featureVectorsTest)




    featureVectorsTrainNormalized = []
    featureVectorsTestNormalized = []


    for vec in featureVectorsTrain:
        vecNormalized = vec/np.linalg.norm(vec)
        featureVectorsTrainNormalized.append(vecNormalized)

    for vec in featureVectorsTest:
        vecNormalized = vec/np.linalg.norm(vec)
        featureVectorsTestNormalized.append(vecNormalized)	

    trainMean = np.mean(featureVectorsTrainNormalized, axis = 0)
    testMean = np.mean(featureVectorsTestNormalized, axis = 0)	

    featureVectorsTrainNormalizedCentered = []
    featureVectorsTestNormalizedCentered = []

    for vec in featureVectorsTrainNormalized:
        vecCentered = vec - trainMean
        featureVectorsTrainNormalizedCentered.append(vecCentered)

    for vec in featureVectorsTestNormalized:
        vecCentered = vec - testMean
        featureVectorsTestNormalizedCentered.append(vecCentered)



    #Train a multiclass SVM and a one-class SVM given all feature vectors
    t1 = time.time()
    modelSVM = svm.SVC(kernel="rbf", C=1e6, probability=True) # little regularization needed - get this training set right, neglect margin
    modelSVM.fit(featureVectorsTrainNormalizedCentered, labelsTrain)
    print 'SVM training took ',(time.time() - t1) ,' seconds'

    t1 = time.time()
    modelSVM_1 = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    modelSVM_1.fit(X_train)
    print 'SVM_1 training took ',(time.time() - t1) ,' seconds'

    countCorrect = 0

    #Test the SVM using the extracted testFeatures
    t1 = time.time()
    prediction = []
    for index in range(len(filesTestNames)):
        features =  np.array(featureVectorsTestNormalizedCentered[index]).reshape((1, -1))
        prediction.append(modelSVM.predict(features))
        print 'For image ', filesTestNames[index], ' ... Predicted: ', prediction[-1], ' TrueLabel: ', labelsTest[index]
        if prediction[-1] == labelsTest[index]:
            countCorrect+=1
    print 'SVM test took ',(time.time() - t1) ,' seconds'
    print 'Accuracy: ', float(countCorrect)/len(labelsTest)

    #confusion matrix
    set_ = {}
    map(set_.__setitem__, labelsTrain, [])
    classes = list(set_.keys())
    cnf_matrix = confusion_matrix(labelsTest, prediction)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=classes)
    plt.show()




    filters = net.params['conv1'][0].data
    vis_square(filters.transpose(0, 2, 3, 1))


    #activatedFeatures = net.blobs['conv1'].data #Not working on activation features....
    #vis_square(activatedFeatures.transpose(0, 2, 3, 1))


if __name__=='__main__':	
    main(sys.argv[1:])
