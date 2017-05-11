	#Open allNamesListFile, and split the data into:

	ImagesTrainSet = "ImagesTrainSet.txt"  
	ImagesTestSet = "ImagesTestSet.txt"

	trainSet = []
	testSet = []

	rangeImageIndices = range(len(ImgSet)) #is the total number of images

	numberTrainSample = trainPercentage*len(ImgSet)    # Extract, for example, 70% of indexes of the images
	random_indexes = random.sample(rangeImageIndices, numberTrainSample) # Randomly extracts 70% of the images indexes
	
	for i in range(numberTrainSample):
		for j in rangeImageIndices:
			if random_indexes(i) == j:
				trainSet[i] = ImgSet(j)
			else:
				testSet[i] = ImgSet(j)

	trainFile = open(ImagesTrainSet, "w")
	trainFile.write("\n".join(trainSet))
	trainFile.close()

	testFile = open(ImagesTestSet, "w")
	testFile.write("\n".join(testSet))
	testFile.close()

	return [trainSet, testSet]