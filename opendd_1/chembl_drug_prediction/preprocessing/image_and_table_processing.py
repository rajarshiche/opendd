# import necessary packages for data loading and processing
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import backend as K

import pandas as pd
import numpy as np

import cv2
import os

import glob
import pathlib
import typing
from os import path
from joblib import Parallel, delayed



def load_table(inputPath): # load_table function accepts the path to the input dataset

    df = pd.read_csv(inputPath, delimiter = "\t", header = None)
    #df = pd.read_csv(inputPath, delimiter = " ", header = None)

    return df


### Serial processing of images
def image_data_extract(imagePaths): # load_table function accepts the path to the input images
    images = []

    for (i, imagepath) in enumerate(imagePaths):

        image = cv2.imread(imagepath)
        image = cv2.resize(image, (224,224),interpolation=cv2.INTER_AREA)  # width followed by height

        image_to_append = image

        # add the resized images to the images list on which the network will be trained
        images.append(image_to_append)
        

    # # return the set of images as an array
    return np.array(images)


def preprocess_img(path: typing.Union[str, pathlib.Path] = "chembl_drug_prediction/input_data/target_images") -> typing.List:
    """open and resize image"""
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    return img        

### Batch processing of images
def preprocess_all_imgs(image_path: typing.Union[str, pathlib.Path] = "chembl_drug_prediction/input_data/target_images") -> typing.List:
    """use pool of available processors to process images to numpy array"""

    image_path = path.abspath(image_path)
    image_paths = sorted(list(glob.glob(path.join(f"{image_path}/*.png"))))
    imgs = Parallel(-1)(delayed(preprocess_img)(img) for img in image_paths)
    return (np.asarray(imgs)).astype("float")
    ##return (np.asarray(imgs) / 255.0).astype("float")


