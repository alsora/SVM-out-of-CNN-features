# SVM-out-of-CNN-features
Python implementation of an SVM on top of a Caffe CNN

To run using pascal voc classes: 

        $ python CNN_SVM_main.py -m prototxt/legacy/yolo_deploy.prototxt -w weights/yolo_deploy.caffemodel -i samples_namesVOC.txt -s voc

    
To run using imagenet classes you have to choose a proper txt file containing the images names (like the one generated using get_path.py) and set the argument -s imagenet
