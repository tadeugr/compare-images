#!/usr/bin/env python3

import sys
import os
import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import h5py
import cv2
import csv
import uuid
import logging
import traceback

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from keras.layers import Flatten, Dense, Input,concatenate
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout
from keras.models import Model
from keras.models import Sequential
sys.stderr = stderr

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from scipy import spatial
from PIL import Image
import imghdr


# pip install "numpy<1.17"



vgg16 = keras.applications.VGG16(weights='imagenet', include_top=True, pooling='max', input_shape=(224, 224, 3))
basemodel = Model(inputs=vgg16.input, outputs=vgg16.get_layer('fc2').output)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
convertMap = {}

def get_feature_vector(img):
    img1 = cv2.resize(img, (224, 224))
    feature_vector = basemodel.predict(img1.reshape(1, 224, 224, 3))
    return feature_vector

def calculate_similarity(vector1, vector2):
    return 1 - spatial.distance.cosine(vector1, vector2)

def read_image(filename):
    oriimg = cv2.imread(filename)
    return oriimg

def gifToPng(infile):
    # Open GIF file
    try:
        im = Image.open(infile)
    except IOError:
        print("Cant load to convert", infile)
        sys.exit(1)
    
    i = 0
    mypalette = im.getpalette()
    prefix = 'auto-gif2png-%s-' % str(uuid.uuid1())
    pngFileNames = []
    try:
        while 1:
            im.putpalette(mypalette)
            new_im = Image.new("RGBA", im.size)
            new_im.paste(im)

            fileName = prefix+str(i)+'.png'
            new_im.save(fileName)
            pngFileNames.append(fileName)

            i += 1
            im.seek(im.tell() + 1)

            # @TODO: compare all layers of a GIF file 
            # Comment out this line to process all
            # GIF layers
            break
    except EOFError:
        pass # end of sequence
    
    # This version is considering
    # only the first GIF layer
    return pngFileNames[0]

def getImageType(imgPath):
    return imghdr.what(imgPath)

def isFileTypeSupported(imgPath, imgType = None):
    supportedFiles = ['jpeg', 'png', 'gif']
    if not imgType:
        imgType = getImageType(imgPath)
    return imgType in supportedFiles

def formatFiles(imgPathArray):
    try:
        for imgPath in imgPathArray:
            imgType = getImageType(imgPath)
            isSupported = isFileTypeSupported(imgPath, imgType)
            if not isSupported:
                raise Exception('Image %r type not supported' % imgPath)

            if imgType == 'gif':
                # Avoid converting the same image
                # more than one time
                if imgPath == getConvertMap(imgPath):
                    pngFile = gifToPng(imgPath)
                    convertMap[imgPath] = pngFile
        return True    
    except Exception as e:
        logger.error('ERROR: '+ str(e))
        traceback.print_exc()
        sys.exit(1)

def getConvertMap(imgPath):
    try:
        return convertMap[imgPath]
    except KeyError:
        return imgPath

def main():
    try:
        '''
        #processImage("tests/img/img-003.gif")
        print(isFileTypeSupported('tests/img/img-003.gif'))

        sys.exit()
        '''

        '''
        img1 = read_image("tests/img/img-001.jpeg")
        img2 = read_image("tests/img/img-001.jpeg")
        #img3 = read_image("tests/img/img-003.gif")
        img3 = read_image("foo0.png")
        f1 = get_feature_vector(img1)
        f2 = get_feature_vector(img2)
        f3 = get_feature_vector(img3)
        print(calculate_similarity(f1, f2))
        print(calculate_similarity(f1, f3))
        sys.exit()
        '''
    

        '''
        for i in range(1,10):
            img1 = read_image("tests/img/img-00"+str(i)+".jpeg")
            f1 = get_feature_vector(img1)
            for j in range(1,10):
                img2 = read_image("tests/img/img-00"+str(j)+".jpeg")
                f2 = get_feature_vector(img2)
                print("tests/img/img-00"+str(i)+".jpeg"+" "+"tests/img/img-00"+str(j)+".jpeg")
                print(calculate_similarity(f1, f2))
        '''

        with open('tests/input.csv') as csvfile:
            readCSV = csv.reader(csvfile, delimiter=';')
            next(readCSV, None)
            for row in readCSV:
                img1Path = row[0]
                img2Path = row[1]

                formatFiles((img1Path, img2Path))
                # Overwrite paths if image was converted
                # In this case, use the temporary file name
                img1Path = getConvertMap(img1Path)
                img2Path = getConvertMap(img2Path)

                logger.info('Comparing %s x %s', img1Path, img2Path)
               
                img1 = read_image(img1Path)
                img2 = read_image(img2Path)
                f1 = get_feature_vector(img1)
                f2 = get_feature_vector(img2)
                print(img1Path+" "+img2Path)
                print(calculate_similarity(f1, f2))
    except Exception as e:
        logger.error('ERROR: '+ str(e))
        traceback.print_exc()


if __name__ == "__main__":
    main()