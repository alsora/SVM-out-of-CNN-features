import xml.etree.ElementTree as ET




def createSamplesDatastructures(images_dir, annotations_dir, interesting_labels, mode):


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
                label = readLabelFromAnnotation(annotationCompletePath, interesting_labels)
                samplesLabels.append(label)

        imagesFolderPath = images_dir
        annotationsFolderPath = annotations_dir


        return [samplesNames, samplesImages, samplesLabels]




def normalizeData(featuresVector):


	featureVectorsNormalized = []

	for vec in featuresVector:
		vecNormalized = vec/np.linalg.norm(vec)
		featureVectorsNormalized.append(vecNormalized)

	mean = np.mean(featureVectorsTrainNormalized, axis = 0)

	featureVectorsNormalizedCentered = []

	for vec in featureVectorsNormalized:
		vecCentered = vec - mean
		featureVectorsNormalizedCentered.append(vecCentered)


	return featureVectorsNormalizedCentered




def readLabelFromAnnotation(annotationFileName, interesting_labels):
    #Parse the given annotation file and read the label

    tree = ET.parse(annotationFileName)
    root = tree.getroot()
    for obj in root.findall('object'):
        label = obj.find("name").text
        if label in interesting_labels:
            return label
        else:
            return 'unknown'
