#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import h5py
import cv2
from keras.layers import Flatten, Dense, Input,concatenate
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout
from keras.models import Model
from keras.models import Sequential
import tensorflow as tf
from scipy import spatial

# pip install "numpy<1.17"

vgg16 = keras.applications.VGG16(weights='imagenet', include_top=True, pooling='max', input_shape=(224, 224, 3))
basemodel = Model(inputs=vgg16.input, outputs=vgg16.get_layer('fc2').output)

def get_feature_vector(img):
    img1 = cv2.resize(img, (224, 224))
    feature_vector = basemodel.predict(img1.reshape(1, 224, 224, 3))
    return feature_vector

def calculate_similarity(vector1, vector2):
    return 1 - spatial.distance.cosine(vector1, vector2)

def read_image(filename):
    oriimg = cv2.imread(filename)
    return oriimg

def main():
    img1 = read_image("./img1.jpg")
    img2 = read_image("./img2.jpg")
    img3 = read_image("./img3.jpg")
    f1 = get_feature_vector(img1)
    f2 = get_feature_vector(img2)
    f3 = get_feature_vector(img3)
    print(calculate_similarity(f1, f2)) # 0.7384121417999268
    print(calculate_similarity(f1, f3)) # 0.48573723435401917



if __name__ == "__main__":
    main()