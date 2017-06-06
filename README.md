
# SVM-out-of-CNN-features
Python implementation of an SVM on top of a Caffe CNN

You need the following dependencies:
- numpy
- sklearn
- cv2
- [pyCocoTools](https://github.com/pdollar/coco)
- [Caffe, pyCaffe](https://github.com/BVLC/caffe)

Create a folder called prototxt containing the .prototxt files of the network you want to use.
Create a folder called weights containing the .caffemodel files of the network you want to use.

NOTE: supported networks are VGG16, RESNET101, GOOGLENET.
Adding a new network is straightforward: simply look at where the -n parameter is used and add a field for the network of interest.

For PascalVOC: create a folder containing the JPEGImages and a folder containing the .xml annotation files.
For MS COCO: create a folder containing the .json annotations file. The images are optional, can be downloaded automatically when executing the main program.

To run use:

    $ python CNN_SVM_detection.py -w weights/VGG_ILSVRC_16_layers.caffemodel -p prototxt/VGG_ILSVRC_16_layers.prototxt -i images -a annotations -t coco -n vggnet -g

This will run the system using the provided VGG16 network. It will download images in the images folder (or it will check if images are already there), it will expect to find the .json file in annotations folder. The parameter "-t" indicates the type of the dataset (it can be coco or voc); the parameter "-n" indicates the network type (it can be vggnet, googlenet, resnet); the parameter "-g" is a flag denoting if we want to use gpu or not (if you don't put -g the program will be run using cpu); there is another parameter "-s" which denotes if you want to use the predefined novelty classifier or if you want to perform a grid search to find the best among oneclassSVM, 2classSVM, isolationTree.
