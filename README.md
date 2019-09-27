# Overview

This project calculates the similarity of images. It reads an input CSV file containing the absolute path of images tuples that must be compared. The result is another CSV file containing the similarity score and the time elapsed for each analysis.

# Methodology

The algorithm uses [Keras](https://keras.io/) and [Tensorflow](https://www.tensorflow.org/).

VGG16 is the chosen model, with weights pre-trained on ImageNet.

That being said, this project prioritizes accuracy with a fairness tradeoff for resource allocation.

# Important notes

* The similarity score is a range from 0 (zero) to 1 (one). Zero being "identical" and One being "completely different". **Note that similarity scores under 30% are being considered "completely different".**

* Supported files: JPEG, PNG and GIF.

* GIF files are converted to PNG RBG format.

* Transparency on PNG files will be neglected.

* All images are rezied to 224x224 (due Keras requirement). If your image set container images much bigger or smaller, there will a slightly accuracy degradation. 

* Libpng 1.6+ is more stringent about checking ICC profiles than previous versions. If your system has Libpng 1.6+ installed, you might see some messages like `libpng warning: iCCP: CRC error`. You can ignore those messages. They will not affect the analysis.

* All images in the tests folder are under GNU license.



virtualenv is recommended

libpng warning: iCCP: CRC error
mogrify *.png
convert in.png out.png

Note for GIF



CSV delimiter must ;

apt install -y python3 python3-pip libsm6 libxext6 libxrender-dev imagemagick

https://www.imgonline.com.ua/eng/similarity-percent.php
https://deepai.org/machine-learning-model/image-similarity

# Input file

The CSV will contain 2 fields (image1 and image2) with N records. Each field contains the absolute path to an image file.

| image1 |  image2 |
|--------|---------|
| aa.png |  ba.png |
| ab.png |  bb.png |
| ac.png |  ac.gif |
| ad.png |  bd.png |

# Output file

The CSV file will need to have 4 fields (image1, image2, similar, elapsed - in seconds) and have the same amount of records as the input file.

The values that fall under the first 2 fields (image1 and image2) need to be the same as the input file.

The values that fall under the similar field will need to represent a "score" based on how similar image1 is to image2

| image1 |  image2 | similar | elapsed  |
|--------|---------| ------- | -------- |
| aa.png |  ba.png | 0       | 0.006    |
| ab.png |  bb.png | 0.23    | 0.843    |
| ac.png |  ac.gif | 0       | 1.43     |
| ad.png |  bd.png | 1       | 2.32     |


# Validation metodolgy

neural network
external tools https://www.imgonline.com.ua/eng/similarity-percent-result.php
visual analysis 
