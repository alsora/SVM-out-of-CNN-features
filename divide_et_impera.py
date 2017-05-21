import xml.etree.cElementTree as ET
from os import walk, mkdir
from os.path import join, isdir
import argparse
import sys
import cv2
from collections import OrderedDict

INTERESTING_LABELS = ['aeroplane', 'bird', 'cat', 'boat', 'horse']

def getBBs(annotations_dict, xml_path):
    for root, dirs, files in walk(ANNOTATIONS_DIR_IN):
        for file in files:
            name, extension = file.split(".")
            if extension == "xml":
                xml_path = join(root, file)
                tree = ET.parse(xml_path)
                root = tree.getroot()
                filename = root.find("filename").text
                obj_number = 0
                for object in root.findall("object"):
                    label = object.find("name").text
                    if label in INTERESTING_LABELS:
                        bndbox = object.find("bndbox")
                        annotations_dict[filename+"_"+str(obj_number)] = {}
                        annotations_dict[filename+"_"+str(obj_number)]["xmin"] = bndbox.find("xmin").text
                        annotations_dict[filename+"_"+str(obj_number)]["ymin"] = bndbox.find("ymin").text
                        annotations_dict[filename+"_"+str(obj_number)]["xmax"] = bndbox.find("xmax").text
                        annotations_dict[filename+"_"+str(obj_number)]["ymax"] = bndbox.find("ymax").text
                        annotations_dict[filename+"_"+str(obj_number)]["label"] = label
                        obj_number += 1
    return


def dumpDictToXMLs(annotations_dict):

    for filename, features in annotations_dict.iteritems():

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
        tree.write(join(ANNOTATIONS_DIR_OUT,filename)+".xml")

    return



def cropImages(annotations_dict):
    for image_path in walk(IMAGES_DIR_IN):
        image_name = image_path.split("/")[-1].split(".")[-1]
        image = cv2.imread(image_path)
        cv2.imshow(image)
        for key, features in annotations_dict.iteritems():
            if image_name in key:
                xmin = features["xmin"]
                ymin = features["ymin"]
                xmax = features["xmax"]
                ymax = features["ymax"]
                cropped_image = image[xmin:xmax, ymin:ymax]
                cv2.imwrite(IMAGES_DIR_OUT+key+".jpg", cropped_image)

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
        ANNOTATIONS_DIR_IN = parser.annotations_dir_in
    elif args.annotations_dir_out:
        ANNOTATIONS_DIR_OUT = parser.annotations_dir_out
    elif args.images_dir:
        IMAGES_DIR_IN = parser.images_dir_in
    elif args.output_dir:
        IMAGES_DIR_OUT = parser.images_dir_out
    else:
        print "Invalid arguments.\nRun \n\t$ python divide_et_impera.py --help\n"
        sys.exit(2)

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









