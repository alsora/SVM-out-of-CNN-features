import sys
sys.path.append("/home/bamos/repos/caffe-local/python")

import argparse
import numpy as np
import os
import time

from caffe.io import array_to_blobproto
from collections import defaultdict
from skimage import io



def compute_mean(image_set, mean_filename):

    if not image_set:

        print("the provided image_set is empty....")

    else:

        mean = np.zeros(image_set[0].shape)
        N = 0

        for img in image_set:
            if img.shape == (mean.shape):
                mean[0] += img[0,:, :]
                mean[1] += img[1,:, :]
                mean[2] += img[2,:, :]
                N += 1


        mean /= N

        blob = array_to_blobproto(mean)
        with open("{}.binaryproto".format(mean_filename), 'wb') as f:
            f.write(blob.SerializeToString())
        np.save("{}.npy".format(mean_filename), mean)

        meanImg = np.transpose(mean.astype(np.uint8), (1, 2, 0))
        io.imsave("{}.png".format(mean_filename), meanImg)

