import xml.etree.cElementTree as ET
from os import walk, mkdir
from os.path import join, isdir
import argparse
import sys
import cv2
from collections import OrderedDict


def getBBs(annotations_dir_in, annotations_dict, interesting_labels):
    for root, dirs, files in walk(annotations_dir_in):
        for file in files:
            name, extension = file.split(".")
            if extension == "xml":
                xml_path = join(root, file)
                tree = ET.parse(xml_path)
                root_xml = tree.getroot()
                #filename = root_xml.find("filename").text
                obj_number = 0
                for object in root_xml.findall("object"):
                    label = object.find("name").text
                    if label in interesting_labels or not interesting_labels:
                        bndbox = object.find("bndbox")
                        annotations_dict[name+"_"+str(obj_number)] = {}
                        annotations_dict[name+"_"+str(obj_number)]["xmin"] = bndbox.find("xmin").text
                        annotations_dict[name+"_"+str(obj_number)]["ymin"] = bndbox.find("ymin").text
                        annotations_dict[name+"_"+str(obj_number)]["xmax"] = bndbox.find("xmax").text
                        annotations_dict[name+"_"+str(obj_number)]["ymax"] = bndbox.find("ymax").text
                        annotations_dict[name+"_"+str(obj_number)]["label"] = label
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



def cropImages(images_dir_in, images_dir_out, annotations_dict):
    for root, dirs, files in walk(images_dir_in):
        for image_name in files:
            name, extension = image_name.split(".")
            if extension == "jpg":
                image_path = join(root, image_name)
                image = cv2.imread(image_path)
                #cv2.imshow("original", image)
                for key, features in annotations_dict.iteritems():
                    if name in key:
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


def extractBBoxesImages(annotations_dir_in,annotations_dir_out,images_dir_in,images_dir_out, interesting_labels ):


    if not isdir(annotations_dir_in) or not isdir(images_dir_in):
        print "The input directories are not valid"
        sys.exit(2)
    if not isdir(annotations_dir_out):
        try:
            mkdir(annotations_dir_out)
        except OSError as e:
            print e
            sys.exit(2)
    if not isdir(images_dir_out):
        try:
            mkdir(images_dir_out)
        except OSError as e:
            print e
            sys.exit(2)

    annotations_dict = {}

    getBBs(annotations_dir_in, annotations_dict,interesting_labels)
    dumpDictToXMLs(images_dir_out, annotations_dir_out, annotations_dict)
    cropImages(images_dir_in, images_dir_out, annotations_dict)









