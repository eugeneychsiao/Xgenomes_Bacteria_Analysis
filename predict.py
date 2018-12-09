from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
from skimage.transform import rescale
import cv2



model = load_model('lambda_t7.h5')
basedir = 'C:/Users/Eugen/Documents/School/CS542/Project/test_data/test.1600.jpg'
# os.chdir(basedir)


img = image.load_img(basedir,target_size=(150,150))
img_tensor = image.img_to_array(img)
img_tensor = (np.expand_dims(img_tensor, axis=0))
img_tensor /= 255.

imgplot = plt.imshow(img)
# plt.show()


result = int(np.round(model.predict(img_tensor)))

if(result == 0):
	print("image is lambdaphage")
if(result == 1):
	print("image is t7 bacteria")