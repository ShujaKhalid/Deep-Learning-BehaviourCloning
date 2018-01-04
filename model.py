### Pull the acquired data from the csv file
import csv
import pandas
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import sklearn
from random import shuffle
import os
import csv

samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('\\')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

with open('./driving_log.csv') as csvfile:
    data = pandas.read_csv(csvfile, header=None)

data_array = np.array(data) # convert to numpy array
center_view = list()

#cen_images = data_array[:,0]
cen_images = glob.glob('./IMG/center*.jpg')
#print(cen_images.shape)
for i in cen_images:
    center_view.append(cv2.imread(i))

center_view = np.array(center_view) # image array ready!
#print(center_view.shape) 

steer_angle = data_array[:,3] # steer-angle array ready!
#print(steer_angle.shape)

# Assign training data
X_train = center_view
y_train = steer_angle

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

### Create and run the model
import keras
from keras.layers import Flatten, Convolution2D, Dense, Activation, Dropout, Lambda, Cropping2D
from keras.models import Sequential

# Using NVIDIA model
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3))) # Normalize and mean center the data
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(3,160,320)))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, subsample=(2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7, verbose=1)
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=7, verbose=1)
model.save('model.h5')
