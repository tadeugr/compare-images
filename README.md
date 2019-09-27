# Overview

This project calculates the similarity of images. It reads an input CSV file containing the absolute path of images tuples that must be compared. The result is another CSV file containing the similarity score and the time elapsed for each analysis.

# Methodology

The algorithm uses [Keras](https://keras.io/) and [Tensorflow](https://www.tensorflow.org/).

VGG16 is the chosen model, with weights pre-trained on ImageNet.

That being said, this project prioritizes accuracy with a fairness tradeoff for resource allocation.

# Important notes

* The similarity score is a range from 0 (zero) to 1 (one). Zero being "identical" and One being "completely different". **Note that similarity scores under 30% are being considered "completely different".**

* When you run the script for the first time, it will download the VGG16 model (about 500M). It might take a while, although it will happen only once. You will see the following message `Downloading data from https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5`

* Supported files: JPEG, PNG and GIF.

* GIF files will be converted to PNG RBG format.

* Transparency on PNG files will be neglected.

* All images are resized to 224x224 (due Keras requirement). If your image set contains images much bigger or smaller, there will a slightly accuracy degradation. 

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

The CSV file will have 4 fields (image1, image2, similar, elapsed - in seconds) and have the same amount of records as the input file.

The values that fall under the first 2 fields (image1 and image2) str the same as the input file.

The values that fall under the similar field will represent a "score" based on how similar image1 is to image2

| image1 |  image2 | similar | elapsed  |
|--------|---------| ------- | -------- |
| aa.png |  ba.png | 0       | 0.006    |
| ab.png |  bb.png | 0.23    | 0.843    |
| ac.png |  ac.gif | 0       | 1.43     |
| ad.png |  bd.png | 1       | 2.32     |


# How to build and run

Python `virtualenv` is highly recommended.

Checkout/download the source code and go to its directory.

## MacOS

Install dependencies:

```
brew install python3
```

Install requirements:

```
pip3 install -r requirements.txt
```

Run the example *(from project's root folder)*:

```
python3 compare-images.py tests/input.csv output.csv
```

## Windows

Install Python3 and Pip3 for Windows. Make sure your PATH env variable is set correctly.

*Binaries python and pip must point to python3 and pip3 respectively*

Install requirements:

```
pip install -r requirements.txt
```

Run the example *(from project's root folder)*:

```
python compare-images.py tests\input.csv output.csv
```

# Ubuntu Linux

*It also works on Docker containers*

Install dependencies:

```
apt update
apt install -y python3 python3-pip libsm6 libxext6 libxrender-dev
```

Install requirements:

```
pip3 install -r requirements.txt
```

Run the example *(from project's root folder)*:

```
python3 compare-images.py tests/input.csv output.csv
```

# Usage

Command usage:

```
python compare-images.py <INPUT CSV PATH> <OUTPUT CSV PATH>
```

Note that you can run the command with CSVs absolute paths. Also, if your input CSV file has image's absolute paths, it is not mandatory to in the project's root folder to run the script.