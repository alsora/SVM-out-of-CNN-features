import numpy as np
import sys, getopt
import cv2
import os
import glob



def writeImages(num,classes):

	imagesFolder = "ImageNet/"
	outputFileName = "samples_namesNET.txt"


	with open (outputFileName,'w') as myfile:
		for animal in classes:

			path = os.path.join(imagesFolder, animal)

			fnames = os.listdir(path)

			if num == 0:
				for eachfile in fnames: myfile.write(path + '/' + eachfile + '\n')
			else:
				for n in range(0,num):
					myfile.write(path + '/' + fnames[n] + '\n')







def main(argv):


	num = 0

	if len(argv) == 1:
		num = int(argv[0])
	else:
		print 'Provide the number of images !!!!!!!', 
		sys.exit(2)
	
	interesting_labels = ['n02117135','n02129604','n02134084','n02398521','n02481103']

	writeImages(num,interesting_labels)



if __name__=='__main__':	
	main(sys.argv[1:])