# XGenomes - Binary Classification of T7 and lambda phage
CS542 Project in Collaboration with XGenomes.

## Getting started
Download all packages below:
- Numpy
- Matplotlib
- scipy
- PIL
- Tensorflow
- Keras

These packages are used to run our ML model.

## Structure
setup.py is a preprocessing file

cnn_model.py is our network architecture

predict.py loads our model and predicts an input image.

## Quickstart
To use our program download the pretrained model from the following link: https://drive.google.com/file/d/1L_NJTtDb9WLfScMmRH0KtgVX529m9dWl/view?usp=sharing

and run predict.py using 
```
python predict.py
```
Within predict.py line 14 must be changed in order to specify which file to validate. 

The line that specifies basedir must be changed to the directory and specific file you would like to validate. An example is shown below.
```
basedir = 'C:/Users/Eugen/Documents/School/CS542/Project/test_data/test.1600.jpg'
```
Please note we are using python3. Please verify your python version.

## Output
This program will output one of two lines into the command line:
```
image is lambdaphage
```
or 
```
image is t7 bacteria
```
