import os, shutil
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import image
from cnn_model import model
import matplotlib.pyplot as plt


# basedir1 = 'C:/Users/Eugen/Documents/School/CS542/Project/xgenomes_processed_data/lambda/'
# basedir2 = 'C:/Users/Eugen/Documents/School/CS542/Project/xgenomes_processed_data/t7/'
# os.chdir(basedir2)
# def main(): 
#     i = 0
      
#     for filename in os.listdir(basedir2): 
#         dst = "t7."+str(i) + ".jpg"
#         src =basedir2+ filename 
#         dst =basedir2+ dst 
#         os.rename(src, dst) 
#         i += 1
   
# if __name__ == '__main__': 
    
#     main() 

base_dir = 'C:/Users/Eugen/Documents/School/CS542/Project/train_data'
original_lambda_dir = 'C:/Users/Eugen/Documents/School/CS542/Project/xgenomes_processed_data/lambda/'
original_t7_dir = 'C:/Users/Eugen/Documents/School/CS542/Project/xgenomes_processed_data/t7/'

train_dir = os.path.join(base_dir, 'train')
# os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
# os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
# os.mkdir(test_dir)

train_lambda_dir = os.path.join(train_dir, 'lambda')
# os.mkdir(train_lambda_dir)
train_t7_dir = os.path.join(train_dir, 't7')
# os.mkdir(train_t7_dir)

validation_lambda_dir = os.path.join(validation_dir, 'lambda')
# os.mkdir(validation_lambda_dir)
validation_t7_dir = os.path.join(validation_dir, 't7')
# os.mkdir(validation_t7_dir)
test_lambda_dir = os.path.join(test_dir, 'lambda')
# os.mkdir(test_lambda_dir)
test_t7_dir = os.path.join(test_dir, 't7')
# os.mkdir(test_t7_dir)


fnames = ['lambda{}.jpg'.format(i) for i in range(3000)]
for fname in fnames:
	src = os.path.join(original_lambda_dir, fname)
	dst = os.path.join(train_lambda_dir, fname)
	shutil.copyfile(src, dst)

fnames = ['lambda{}.jpg'.format(i) for i in range(3000, 4000)]
for fname in fnames:
	src = os.path.join(original_lambda_dir, fname)
	dst = os.path.join(validation_lambda_dir, fname)
	shutil.copyfile(src, dst)

fnames = ['lambda{}.jpg'.format(i) for i in range(4000, 5000)]
for fname in fnames:
	src = os.path.join(original_lambda_dir, fname)
	dst = os.path.join(test_lambda_dir, fname)
	shutil.copyfile(src, dst)

fnames = ['t7.{}.jpg'.format(i) for i in range(3000)]
for fname in fnames:
	src = os.path.join(original_t7_dir, fname)
	dst = os.path.join(train_t7_dir, fname)
	shutil.copyfile(src, dst)

fnames = ['t7.{}.jpg'.format(i) for i in range(3000, 4000)]
for fname in fnames:
	src = os.path.join(original_t7_dir, fname)
	dst = os.path.join(validation_t7_dir, fname)
	shutil.copyfile(src, dst)

fnames = ['t7.{}.jpg'.format(i) for i in range(4000, 5000)]
for fname in fnames:
	src = os.path.join(original_t7_dir, fname)
	dst = os.path.join(test_t7_dir, fname)
	shutil.copyfile(src, dst)

train_datagen = ImageDataGenerator(rescale=1./255,
								   rotation_range= 40,
								   width_shift_range= 0.2,
								   height_shift_range= 0.2,
								   shear_range= 0.2,
								   zoom_range= 0.2,
								   horizontal_flip=True)
# train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(train_dir,
													target_size=(150,150),
													batch_size=32,
													class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_dir,
														target_size=(150,150),
														batch_size=32,
														class_mode='binary')

history = model.fit_generator(train_generator,
							  steps_per_epoch= 100,
							  epochs= 1,
							  validation_data= validation_generator,
							  validation_steps= 50)

# model.save('lambda_t7.h5')
