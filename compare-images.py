#!/usr/bin/env python3

import sys
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import h5py
import cv2
import csv
import uuid
import logging
import traceback
import time

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
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

def getFeatureVector(img):
    img1 = cv2.resize(img, (224, 224))
    feature_vector = basemodel.predict(img1.reshape(1, 224, 224, 3))
    return feature_vector

def calculateCosineDistance(vector1, vector2):
    return 1 - spatial.distance.cosine(vector1, vector2)

def calculateSimilarity(img1Path, img2Path):
    img1 = readImage(img1Path)
    img2 = readImage(img2Path)
    f1 = getFeatureVector(img1)
    f2 = getFeatureVector(img2)
    return calculateCosineDistance(f1, f2)

def formatNumber2d(number):
    return float("{:.2f}".format(number))

def formatNumberTrailingZero(number):
    return float("{0:g}".format(number))

def getScore(originalScore):
    score = 1 - originalScore
    score = "{:.2f}".format(score)
    score = "{0:g}".format(float(score))
    return score

def readImage(filename):
    img = cv2.imread(filename)
    return img

def gifToPng(infile):
    # Open GIF file
    try:
        im = Image.open(infile)
    except IOError:
        print("Cant load to convert", infile)
        sys.exit(1)
    
    i = 0
    myPalette = im.getpalette()
    prefix = 'auto-converted-gif2png-%s-' % str(uuid.uuid1())
    pngFileNames = []
    try:
        while 1:
            im.putpalette(myPalette)
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

def saveFile(path, content):
    try:
        f = open(path, 'w' )
        f.write(content)
        f.close()
    except Exception as e:
        logger.error('ERROR: '+ str(e))
        traceback.print_exc()

def cleanUp():
    try:
        fileList = glob.glob('auto-converted-*', recursive=True)
        for filePath in fileList:
                os.remove(filePath)
    except Exception as e:
        logger.error('ERROR: '+ str(e))
        traceback.print_exc()


def main():
    try:
        
        if not sys.argv[1]:
            raise Exception('CSV input path not defined')

        csvInputPath = sys.argv[1]

        with open(csvInputPath) as csvfile:
            inputCsv = csv.reader(csvfile, delimiter=';')
            outputCsv = "image1;image2;similar;elapsed\n"
            next(inputCsv, None)
            for row in inputCsv:
                img1PathOriginal = row[0]
                img2PathOriginal = row[1]

                formatFiles((img1PathOriginal, img2PathOriginal))
                # Get paths if image was converted
                # In this case, use the temporary file name
                img1Path = getConvertMap(img1PathOriginal)
                img2Path = getConvertMap(img2PathOriginal)

                logger.info('Comparing %s x %s', img1PathOriginal, img2PathOriginal)

                startTime = time.time()
                similarity = calculateSimilarity(img1Path, img2Path)
                score = getScore(similarity)
                endTime = time.time()
                deltaTime = formatNumber2d(endTime - startTime)
                logger.info('Time elapsed (seconds): %s', str(deltaTime))

                outputCsv += ("%s;%s;" % (img1PathOriginal, img2PathOriginal))
                outputCsv += ("%s;%s\n" % (score, deltaTime))
                print(outputCsv)

                #break
        print(outputCsv)
        cleanUp()

    except Exception as e:
        logger.error('ERROR: '+ str(e))
        traceback.print_exc()

if __name__ == "__main__":
    main()