#!/usr/bin/env python3

#
# Script to calculate the similarity of N pairs of images.
# The scripts reads an input CSV file containing the
# absolute path of the images that will be compared.
# 
# The result is another CSV file containing the similarity score
# and the time elapsed for each analysis. 
#

#
# START import libs
#
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
from scipy import spatial
from PIL import Image
import imghdr

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

#
# END import libs
#

# Common variables

vgg16 = keras.applications.VGG16(weights='imagenet', include_top=True, pooling='max', input_shape=(224, 224, 3))
basemodel = Model(inputs=vgg16.input, outputs=vgg16.get_layer('fc2').output)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
convertMap = {}

# Resize and reshape image
def getFeatureVector(img):
    img1 = cv2.resize(img, (224, 224))
    feature_vector = basemodel.predict(img1.reshape(1, 224, 224, 3))
    return feature_vector

# Calculate vectors cosine distance
def calculateCosineDistance(vector1, vector2):
    return 1 - spatial.distance.cosine(vector1, vector2)

# Calculate similarity based on vector analysis
def calculateSimilarity(img1Path, img2Path):
    img1 = readImage(img1Path)
    img2 = readImage(img2Path)
    f1 = getFeatureVector(img1)
    f2 = getFeatureVector(img2)
    return calculateCosineDistance(f1, f2)

# Format a number with 2 decimal digits
def formatNumber2d(number):
    return float("{:.2f}".format(number))

# Remove trailing zeroes
def formatNumberTrailingZero(number):
    return float("{0:g}".format(number))

# Calculate the final similarity score
def getScore(originalScore):
    # If similarity level is lower than 30%
    # set score to "completely diffrent"
    if originalScore < 0.3:
        originalScore = 0
    
    score = 1 - originalScore
    score = "{:.2f}".format(score)
    score = "{0:g}".format(float(score))
    return score

# Open and read an image
def readImage(filename):
    img = cv2.imread(filename)
    return img

# Convert GIF to PNG
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
            newIm = Image.new("RGBA", im.size)
            newIm.paste(im)

            fileName = prefix+str(i)+'.png'
            newIm.save(fileName)
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

# Get image type (jpeg, png, gif)
def getImageType(imgPath):
    return imghdr.what(imgPath)

# Check supported files formart
def isFileTypeSupported(imgPath, imgType = None):
    supportedFiles = ['jpeg', 'png', 'gif']
    if not imgType:
        imgType = getImageType(imgPath)
    return imgType in supportedFiles

# Analyse image supported types
# Convert GIF to PNG
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

# Check if an image was converted
def getConvertMap(imgPath):
    try:
        return convertMap[imgPath]
    except KeyError:
        return imgPath

# Write a file
def saveFile(path, content):
    try:
        f = open(path, 'w' )
        f.write(content)
        f.close()
    except Exception as e:
        logger.error('ERROR: '+ str(e))
        traceback.print_exc()

# Delete converted files
def cleanUp():
    try:
        fileList = glob.glob('auto-converted-*', recursive=True)
        for filePath in fileList:
                os.remove(filePath)
    except Exception as e:
        logger.error('ERROR: '+ str(e))
        traceback.print_exc()

# Main function
def main():
    logger.debug('Start funcion main')
    try:
        logger.debug('Validating parameters')
        if not sys.argv[1]:
            raise Exception('Input CSV path not defined')
        if not sys.argv[2]:
            raise Exception('Output CSV path not defined')

        csvInputPath = sys.argv[1]
        csvOutputPath = sys.argv[2]
    except IndexError:
        logger.error('ERROR: Required parameter not set')
        logger.error('Usage: python compare-images.py <CSV INPUT FILE PATH> <CSV OUTPUT FILE PATH>')
        logger.error('Usage example: python compare-images.py input.csv output.csv')
        sys.exit(1)
    
    try:
        logger.debug('Opening input CSV')
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

                outputCsvLine = ''
                outputCsvLine += ("%s;%s;" % (img1PathOriginal, img2PathOriginal))
                outputCsvLine += ("%s;%s\n" % (score, deltaTime))
                outputCsv += outputCsvLine
                logger.debug('Result %s', outputCsvLine)

        saveFile(csvOutputPath, outputCsv)
        cleanUp()

        logger.debug('End funcion main')
        logger.info('Results saved on %s', csvOutputPath)

    except Exception as e:
        logger.error('ERROR: '+ str(e))
        traceback.print_exc()

if __name__ == "__main__":
    main()