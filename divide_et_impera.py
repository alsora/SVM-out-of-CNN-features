import xml.etree.cElementTree as ET
from os import walk, mkdir
from os.path import join, isdir
import argparse
import sys
import cv2
from collections import OrderedDict

INTERESTING_LABELS = ['aeroplane', 'bird', 'cat', 'boat', 'horse']

def getBBs(annotations_dict):
    for root, dirs, files in walk(ANNOTATIONS_DIR_IN):
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
                    if label in INTERESTING_LABELS:
                        bndbox = object.find("bndbox")
                        annotations_dict[name+"_"+str(obj_number)] = {}
                        annotations_dict[name+"_"+str(obj_number)]["xmin"] = bndbox.find("xmin").text
                        annotations_dict[name+"_"+str(obj_number)]["ymin"] = bndbox.find("ymin").text
                        annotations_dict[name+"_"+str(obj_number)]["xmax"] = bndbox.find("xmax").text
                        annotations_dict[name+"_"+str(obj_number)]["ymax"] = bndbox.find("ymax").text
                        annotations_dict[name+"_"+str(obj_number)]["label"] = label
                        obj_number += 1
    return


def dumpDictToXMLs(annotations_dict):

    for filename, features in annotations_dict.iteritems():

        #image_name = filename.split(".")[0]
        root = ET.Element("annotation")

        ET.SubElement(root, "folder").text = IMAGES_DIR_OUT
        ET.SubElement(root, "filename").text = filename

        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = features["label"]
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = features["xmin"]
        ET.SubElement(bndbox, "ymin").text = features["ymin"]
        ET.SubElement(bndbox, "xmax").text = features["xmax"]
        ET.SubElement(bndbox, "ymax").text = features["ymax"]

        tree = ET.ElementTree(root)
        tree.write(join(ANNOTATIONS_DIR_OUT, filename)+".xml")

    return



def cropImages(annotations_dict):
    for root, dirs, files in walk(IMAGES_DIR_IN):
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
                        out_path = join(IMAGES_DIR_OUT, key+".jpg")
                        cv2.imwrite(out_path, cropped_image)

    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("annotations_dir_in", type=str,
                        help="the annotation directory input")
    parser.add_argument("annotations_dir_out", type=str,
                        help="the annotation directory output")
    parser.add_argument("images_dir_in", type=str,
                        help="the database of images")
    parser.add_argument("images_dir_out", type=str,
                        help="the output directory for the cropped images")
    args = parser.parse_args()

    if args.annotations_dir_in:
        ANNOTATIONS_DIR_IN = args.annotations_dir_in
    if args.annotations_dir_out:
        ANNOTATIONS_DIR_OUT = args.annotations_dir_out
    if args.images_dir_in:
        IMAGES_DIR_IN = args.images_dir_in
    if args.images_dir_out:
        IMAGES_DIR_OUT = args.images_dir_out


    if not isdir(ANNOTATIONS_DIR_IN) or not isdir(IMAGES_DIR_IN):
        print "The input directories are not valid"
        sys.exit(2)
    if not isdir(ANNOTATIONS_DIR_OUT):
        try:
            mkdir(ANNOTATIONS_DIR_OUT)
        except OSError as e:
            print e
            sys.exit(2)
    if not isdir(IMAGES_DIR_OUT):
        try:
            mkdir(IMAGES_DIR_OUT)
        except OSError as e:
            print e
            sys.exit(2)

    annotations_dict = {}

    getBBs(annotations_dict)
    dumpDictToXMLs(annotations_dict)
    cropImages(annotations_dict)









