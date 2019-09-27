# Overview

This project calculates the similarity of images. It reads an input CSV file containing the absolute path of images tuples that must be compared. The result is another CSV file containing the similarity score and the time elapsed for each analysis.

# Methodology

The algorithm uses [Keras](https://keras.io/) and [Tensorflow](https://www.tensorflow.org/).

VGG16 is the chosen model, with weights pre-trained on ImageNet.

That being said, this project prioritizes accuracy with a fairness tradeoff for resource allocation.

# Important notes

* The similarity score is a range from 0 (zero) to 1 (one). Zero being "identical" and One being "completely different". **Note that similarity scores under 30% are being considered "completely different".**

* When you run the script for the first time, it will download the VGG16 model (about 500M). It might take a while, although it will happen only once.

* Supported files: JPEG, PNG and GIF.

* GIF files will be converted to PNG RBG format.

* Transparency on PNG files will be neglected.

* All images are rezied to 224x224 (due Keras requirement). If your image set container images much bigger or smaller, there will a slightly accuracy degradation. 

* Libpng 1.6+ is more stringent about checking ICC profiles than previous versions. If your system has Libpng 1.6+ installed, you might see some messages like `libpng warning: iCCP: CRC error`. You can ignore those messages. They will not affect the analysis.

* All images in the tests folder are under GNU license.

# Input file

The CSV will contain 2 fields (image1 and image2) with N records. Each field contains the absolute path to an image file.

| image1 |  image2 |
|--------|---------|
| aa.png |  ba.png |
| ab.png |  bb.png |
| ac.png |  ac.gif |
| ad.png |  bd.png |

**The CSV delimiter must be ; (semicolon)**

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


# How to build this project

## Requirements

* Python3 and Pip3

Python `virtualenv` is highly recommended.

## Build

Checkout the source code, go to its folder and run:

```
pip install -r requirements.txt 
```

# How to run this project

Command usage:

```
python compare-images.py <INPUT CSV PATH> <OUTPUT CSV PATH>
```

Example:

```
python compare-images.py my-dataset.csv report.csv
```

# How to run the tests as an example

```
python compare-images.py tests\input.csv output.csv
```

You will be able to find the results in the `output.csv`.