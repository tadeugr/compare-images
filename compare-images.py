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
from keras.layers import Flatten, Dense, Input,concatenate
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout
from keras.models import Model
from keras.models import Sequential
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from scipy import spatial
from PIL import Image


# pip install "numpy<1.17"

vgg16 = keras.applications.VGG16(weights='imagenet', include_top=True, pooling='max', input_shape=(224, 224, 3))
basemodel = Model(inputs=vgg16.input, outputs=vgg16.get_layer('fc2').output)

def get_feature_vector(img):
    height, width, channels = img.shape 
    print(height)
    sys.exit()
    img1 = cv2.resize(img, (224, 224))
    feature_vector = basemodel.predict(img1.reshape(1, 224, 224, 3))
    return feature_vector

def calculate_similarity(vector1, vector2):
    return 1 - spatial.distance.cosine(vector1, vector2)

def read_image(filename):
    oriimg = cv2.imread(filename)
    print(filename)
    print(oriimg)
    sys.exit()
    return oriimg

def processImage(infile):
    try:
        im = Image.open(infile)
    except IOError:
        print("Cant load", infile)
        sys.exit(1)
    i = 0
    mypalette = im.getpalette()

    try:
        while 1:
            im.putpalette(mypalette)
            new_im = Image.new("RGBA", im.size)
            new_im.paste(im)
            new_im.save('foo'+str(i)+'.png')

            i += 1
            im.seek(im.tell() + 1)

    except EOFError:
        pass # end of sequence

def main():

    processImage("tests/img/img-003.gif")

    #img1 = read_image("tests/img/img-001.jpeg")
    #img2 = read_image("tests/img/img-001.jpeg")
    img3 = read_image("tests/img/img-003.gif")
    #f1 = get_feature_vector(img1)
    #f2 = get_feature_vector(img2)
    f3 = get_feature_vector(img3)
    sys.exit()
    print(calculate_similarity(f1, f2)) # 0.7384121417999268
    print(calculate_similarity(f1, f3)) # 0.48573723435401917
    sys.exit()
  

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
            img1 = read_image(img1Path)
            img2 = read_image(img2Path)
            f1 = get_feature_vector(img1)
            f2 = get_feature_vector(img2)
            print(img1Path+" "+img2Path)
            print(calculate_similarity(f1, f2))


if __name__ == "__main__":
    main()