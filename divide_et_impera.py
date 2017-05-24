import xml.etree.cElementTree as ET
from os import walk, mkdir
from os.path import join, isdir
import argparse
import sys
import cv2
from collections import OrderedDict
import random 
import math
from pycocotools.coco import COCO

def splitTrainTest(annotations_dir, interesting_labels, percentage):
    
    #images = "all_images.txt"
    images = []

    #with open (images,'w') as myfile:
    for root, dirs, files in walk(annotations_dir):
        for file in files:
            name, extension = file.split(".")
            if extension == "xml":
                xml_path = join(root, file)
                tree = ET.parse(xml_path)
                root_xml = tree.getroot()

                for obj in root_xml.findall("object"):
                    label = obj.find("name").text
                    if label in interesting_labels:
                        images.append(name)
                        break

            elif extension == 'json':
                jsonAnnotationPath = join(root,file)

                coco = COCO(annotation_file=jsonAnnotationPath)
                catIds = coco.getCatIds(catNms=interesting_labels)



                images = coco.getImgIds(catIds=catIds)




    numElements = len(images)
    rangeImageIndices = range(numElements)
    numberTrainSample = int(round(percentage*numElements))   
    trainIndices = random.sample(rangeImageIndices, numberTrainSample) 

    trainSet = []
    testSet = []

    for i in range(numElements):
        if i in trainIndices:
            trainSet.append(images[i])
        else:
            testSet.append(images[i])

    return trainSet, testSet





        


def getBBs(imagesSet, annotations_dir_in, annotations_dict, interesting_labels, mode):
  

    if mode is 'voc':

        for image in imagesSet:
            path = annotations_dir_in + '/' + image + '.xml'
            tree = ET.parse(path)
            root_xml = tree.getroot()
            #filename = root_xml.find("filename").text
            obj_number = 0
            for object in root_xml.findall("object"):
                label = object.find("name").text
                if label in interesting_labels or not interesting_labels:
                    bndbox = object.find("bndbox")
                    annotations_dict[image+"_"+str(obj_number)] = {}
                    annotations_dict[image+"_"+str(obj_number)]["xmin"] = bndbox.find("xmin").text
                    annotations_dict[image+"_"+str(obj_number)]["ymin"] = bndbox.find("ymin").text
                    annotations_dict[image+"_"+str(obj_number)]["xmax"] = bndbox.find("xmax").text
                    annotations_dict[image+"_"+str(obj_number)]["ymax"] = bndbox.find("ymax").text
                    annotations_dict[image+"_"+str(obj_number)]["label"] = label
                    obj_number += 1
        return

    elif mode is 'coco':
        for root, dirs, files in walk(annotations_dir_in):
            for file in files:
                name, extension = file.split(".")

                if extension == 'json':

                    jsonAnnotationPath = join(root, file)
                    coco = COCO(annotation_file=jsonAnnotationPath)

                    images = coco.loadImgs(ids=imagesSet)


                    for idx, image in enumerate(images):
                        annIds = coco.getAnnIds(imgIds=image['id'])
                        annotations = coco.loadAnns(ids=annIds)
                        obj_number = 0
                        for annotation in annotations:

                           
                            filename = str(images[idx]['file_name'])
                            bndbox = annotation['bbox']
                            image = str(annotation['image_id'])
                            label = str(annotation['category_id'])
                            annotations_dict[image+"_"+str(obj_number)] = {}
                            annotations_dict[image+"_"+str(obj_number)]["xmin"] = str(int(round(bndbox[0])))
                            annotations_dict[image+"_"+str(obj_number)]["ymin"] = str(int(round(bndbox[1])))
                            annotations_dict[image+"_"+str(obj_number)]["xmax"] = str(int(round(bndbox[0] + bndbox[2])))
                            annotations_dict[image+"_"+str(obj_number)]["ymax"] = str(int(round(bndbox[0] + bndbox[3])))
                            annotations_dict[image+"_"+str(obj_number)]["label"] = label
                            annotations_dict[image+"_"+str(obj_number)]["file_name"] = filename
                            obj_number += 1


                return 





def dumpDictToXMLs(images_dir_out, annotations_dir_out, annotations_dict):

    for filename, features in annotations_dict.iteritems():
        #image_name = filename.split(".")[0]
        root = ET.Element("annotation")

        ET.SubElement(root, "folder").text = images_dir_out
        ET.SubElement(root, "filename").text = filename

        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = features["label"]
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = features["xmin"]
        ET.SubElement(bndbox, "ymin").text = features["ymin"]
        ET.SubElement(bndbox, "xmax").text = features["xmax"]
        ET.SubElement(bndbox, "ymax").text = features["ymax"]

        tree = ET.ElementTree(root)
        tree.write(join(annotations_dir_out, filename)+".xml")

    return



def cropImages(imagesSet, image_dir_in, images_dir_out, annotations_dict, mode):

    if mode is 'voc':

        for image_name in imagesSet:
            image_path = image_name + ".jpg"
            image_path = join(image_dir_in, image_path)
            image = cv2.imread(image_path)
            #cv2.imshow("original", image)
            for key, features in annotations_dict.iteritems():
                if image_name in key:
                    xmin = int(features["xmin"])
                    ymin = int(features["ymin"])
                    xmax = int(features["xmax"])
                    ymax = int(features["ymax"])
                    #cv2.waitKey(0)
                    cropped_image = image[ymin:ymax, xmin:xmax]
                    #cv2.imshow("cropped", cropped_image)
                    #cv2.waitKey(0)
                    out_path = join(images_dir_out, key+".jpg")
                    cv2.imwrite(out_path, cropped_image)

        return


    elif mode is 'coco':

        for key, features in annotations_dict.iteritems():
            
            image_path = features['file_name']
            image_path = join(image_dir_in, image_path)
            image = cv2.imread(image_path)

            xmin = features["xmin"]
            ymin = features["ymin"]
            xmax = features["xmax"]
            ymax = features["ymax"]
            cropped_image = image[ymin:ymax, xmin:xmax]

            out_path = join(images_dir_out, key+".jpg")
            cv2.imwrite(out_path, cropped_image)



        return




                    






def extractBBoxesImages(imagesSet, images_dir_in, annotations_dir_in, images_dir_out, annotations_dir_out, mode, interesting_labels = []):


    #if not isdir(annotations_dir_in) or not isdir(images_dir_in):
    #    print "The input directories are not valid"
    #    sys.exit(2)
    if not isdir(annotations_dir_out) or not isdir(images_dir_out):
        try:
            mkdir(annotations_dir_out)
            mkdir(images_dir_out)
        except OSError as e:
            print e
            sys.exit(2)

        annotations_dict = {}

        getBBs(imagesSet, annotations_dir_in, annotations_dict, interesting_labels, mode)
        dumpDictToXMLs(images_dir_out, annotations_dir_out, annotations_dict)
        cropImages(imagesSet, images_dir_in, images_dir_out, annotations_dict, mode)
    
    else:
        return



