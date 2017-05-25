import xml.etree.cElementTree as ET
from os import walk, mkdir, remove, stat, listdir
from os.path import join, isdir
import sys
import cv2
import random
from pycocotools.coco import COCO
import re


class ImageCropper:
    def __init__(self,image_dir_in, annotations_dir_in, image_dir_out_train, annotations_dir_out_train, image_dir_out_test, annotations_dir_out_test, mode):

        self.image_dir_in = image_dir_in
        self.annotations_dir_in = annotations_dir_in
        self.image_dir_out_train = image_dir_out_train
        self.annotations_dir_out_train = annotations_dir_out_train
        self.image_dir_out_test = image_dir_out_test
        self.annotations_dir_out_test = annotations_dir_out_test

        self.mode = mode

        if self.mode == 'coco':
            for root, dirs, files in walk(self.annotations_dir_in):
                    for file in files:
                        name, extension = file.split(".")
                        if extension == 'json':
                            jsonAnnotationPath = join(root, file)
                            self.coco = COCO(jsonAnnotationPath)
                            break
                    if jsonAnnotationPath:
                        break

    def downloadCoco(self, interestingLabels, maxNumIm = 100):

        catIds = self.coco.getCatIds(catNms=interestingLabels)
        imgIds = []

        for label in catIds:
            img = self.coco.getImgIds(catIds=[label])
            try:
                img = img[:maxNumIm]
            except:
                print label, len(img)
            imgIds.extend(img)

        self.coco.download(tarDir = self.image_dir_in, imgIds=imgIds)

    def splitTrainTest(self, interesting_labels, percentage):
        
        images = []

        for root, dirs, files in walk(self.annotations_dir_in):
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
                    catIds = self.coco.getCatIds(catNms=interesting_labels)
                    images = []
                    for catId in catIds:
                        newImages = self.coco.getImgIds(catIds=catId)
                        images.extend(newImages)

                    matchedIDs = []

                    for root, dirs, files in walk(self.image_dir_in):
                        for file in files:
                            file = file.split('_')[-1]

                            matcher = re.search('(?<=0)([0-9]*)',file)

                            matcher2 = re.search('0*', matcher.group(0))

                            matched = matcher.group(0)[len(matcher2.group(0)) :]

                            matchedIDs.append(matched)

                    images = [x for x in images if str(x) in matchedIDs]

                    break

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

    def getBBs(self, imagesSet, annotations_dict, interesting_labels):

        if self.mode == 'voc':

            for image in imagesSet:
                path = self.annotations_dir_in + '/' + image + '.xml'
                tree = ET.parse(path)
                root_xml = tree.getroot()

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

        elif self.mode == 'coco':

            annotationsIDs = self.coco.getAnnIds(imgIds=imagesSet)
            annotations = self.coco.loadAnns(ids=annotationsIDs)

            for annotation in annotations:
                obj_number = 0
                bndbox = annotation['bbox']

                imageID = str(annotation['image_id'])
                label = str(annotation['category_id'])

                annotations_dict[imageID+"_"+str(obj_number)] = {}
                annotations_dict[imageID+"_"+str(obj_number)]["xmin"] = str(int(round(bndbox[0])))
                annotations_dict[imageID+"_"+str(obj_number)]["ymin"] = str(int(round(bndbox[1])))
                annotations_dict[imageID+"_"+str(obj_number)]["xmax"] = str(int(round(bndbox[0] + bndbox[2])))
                annotations_dict[imageID+"_"+str(obj_number)]["ymax"] = str(int(round(bndbox[0] + bndbox[3])))
                annotations_dict[imageID+"_"+str(obj_number)]["label"] = label

                obj_number += 1

            return




    def dumpDictToXMLs(self, annotations_dict, images_dir_out, annotations_dir_out):

        for filename, features in annotations_dict.iteritems():
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

    def cropImages(self, imagesSet, annotations_dict, images_dir_out):

        if self.mode == 'voc':

            for image_name in imagesSet:
                image_path = image_name + ".jpg"
                image_path = join(self.image_dir_in, image_path)
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

        elif self.mode == 'coco':

            for key, features in annotations_dict.iteritems():
                image_id = int(key.split('_')[0])
                image = self.coco.loadImgs(ids=image_id)
                image_file_name = image[0]['file_name']
                image_path = join(self.image_dir_in, image_file_name)
                image = cv2.imread(image_path)

                xmin = int(features["xmin"])
                ymin = int(features["ymin"])
                xmax = int(features["xmax"])
                ymax = int(features["ymax"])

                cropped_image = image[ymin:ymax, xmin:xmax]
                out_path = join(images_dir_out, key+".jpg")
                cv2.imwrite(out_path, cropped_image)

            return

    def filterImages(self, images_dir_out, annotations_dir_out):

        for file in listdir(images_dir_out):
            filePath = join(images_dir_out, file)
            if stat(filePath).st_size < 10000:
                remove(filePath)
                idName = file.split('.')[0]
                xmlPath = join(annotations_dir_out, idName + '.xml')
                remove(xmlPath)
        return

    def getCocoCategoriesId(self, interesting_labels):
        
        if self.mode == 'coco':
            catIds = self.coco.getCatIds(catNms=interesting_labels)
            catIds = [str(x) for x in catIds]
            return catIds

        else:
            print 'YOU ARE NOT USING COCO MODE!!!!!'
            sys.exit(2)

    def extractBBoxesImages(self, imagesSet, mode, interesting_labels=[]):

        if mode == 'train':
            images_dir_out = self.image_dir_out_train
            annotations_dir_out = self.annotations_dir_out_train
        elif mode == 'test':
            images_dir_out = self.image_dir_out_test
            annotations_dir_out = self.annotations_dir_out_test
        else:
            print "WRONG MODE: YOU HAVE TO USE train or test"
            sys.exit(2)            

        if not isdir(annotations_dir_out) or not isdir(images_dir_out):
            try:
                mkdir(annotations_dir_out)
                mkdir(images_dir_out)
            except OSError as e:
                print e
                sys.exit(2)

            annotations_dict = {}
            self.getBBs(imagesSet, annotations_dict, interesting_labels)
            self.dumpDictToXMLs(annotations_dict, images_dir_out, annotations_dir_out)
            self.cropImages(imagesSet, annotations_dict, images_dir_out)
            self.filterImages(images_dir_out, annotations_dir_out)
            return

        else:
            print 'Both output folders already exist'
            return



