from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt
batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150


# plotting function taken from tutorial but doesn't seem to work well. 
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 4, figsize=(10,10))
    # axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def dataprep(train_dir, validation_dir):


	train_image_generator = ImageDataGenerator(rescale = 1./255, horizontal_flip = True,vertical_flip = True, rotation_range = 360, fill_mode = 'reflect')
	validation_image_generator = ImageDataGenerator(rescale = 1./255)

	# flow from directory loads images from disk / rescales / resizes images

	train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH))
	sample_training_images , _ = next(train_data_gen)
	plotImages(sample_training_images[:5])
	# print(train_data_gen)

	# plotImages(train_data_gen[0][0])
	
	val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH))


	print("done data-prepping")

	return(train_data_gen, val_data_gen)

def compileModel():
	model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

	model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

	model.summary()

	return model


def trainModel(model, train_data_gen):
	
	history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

def saveModel(model):
	print("model should be saved here")


def main():

	PATH = "../Rock_Hold_Dataset"
	train_dir = os.path.join(PATH, 'train')
	validation_dir = os.path.join(PATH, 'validation')
	# get number of datapoints in set

	num_rock_holds = len(os.listdir(PATH))
	print(num_rock_holds)
	print("looking at " , num_rock_holds, " images")

	train_data_gen, val_data_gen = dataprep(train_dir, validation_dir)

	model = compileModel()

	print("train the model here when ready")


	print("done main")
	

if __name__ == '__main__':
    main()