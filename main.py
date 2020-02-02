from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt


def main():
	PATH = "../Rock_Hold_Dataset"
	# get number of datapoints in set

	num_rock_holds = len(os.listdir(PATH))
	print(num_rock_holds)

	print("done main")
	

if __name__ == '__main__':
    main()