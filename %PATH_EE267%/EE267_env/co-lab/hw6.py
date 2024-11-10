from google.colab import drive
drive.mount('/content/gdrive')

#***Preprocessing***

import os
import cv2
from sklearn.utils import shuffle
import numpy as np

%cd /content/gdrive/MyDrive/animal_dataset

# function to load images and respective labels
def load_img(directory):
    Images = []
    Labels = []
    label = 0

for labels in os.listdir(directory):
    if labels == 'butterfly':
        label = 0
    elif labels == 'cat':
        label = 1
    elif labels == 'chicken':
        label = 2
    elif labels == 'elephant':
        label = 3
    elif labels == 'horse':
        label = 4
    elif labels == 'spider':
        label = 5
for image_file in os.listdir(directory + labels):
    image = cv2.imread(directory + labels + '/' + image_file)

    image = cv2.resize(image, (128, 128))

    Images.append(image)
    Labels.append(label)

return shuffle(Images, Labels, random_state=40)

def get_classlabel(class_code):
    labels = {0: 'butterfly',
              1: 'cat',
              2: 'chicken',
              3: 'elephant',
              4: 'horse',
              5: 'spider'}
    return labels[class_code]

Img_data, Label_data = load_img('/content/gdrive/MyDrive/animal_dataset/')
# raw image data from g-drive

Img_np = np.array(Img_data)

Label_np = np.array(Label_data)

print(Img_np.shape)
print(Label_np.shape)

np.save('/content/gdrive/MyDrive/animal_dataset_numpy/Images.npy', Img_np)  # save raw data as numpy array for faster load
np.save('/content/gdrive/MyDrive/animal_dataset_numpy/Labels.npy', Label_np)

loaded_images = np.load('/content/gdrive/MyDrive/animal_dataset_numpy/Images.npy')
loaded_labels = np.load('/content/gdrive/MyDrive/animal_dataset_numpy/Labels.npy')

from random import randint
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 6)
for i in range(0, 6):
    rnd_number = randint(0, len(loaded_labels))
    ax[i].imshow(loaded_images[rnd_number])
    ax[i].set_title(get_classlabel(loaded_labels[rnd_number]))

label_values, count = np.unique(loaded_labels, return_counts=True)
distribution = dict(zip(label_values, count))

plt.bar(list(distribution.keys()), distribution.values(), width=0.6)

plt.xlabel('Image Labels')
plt.ylabel('Count')
plt.show()

print(distribution)

print(loaded_images.shape)
print(loaded_labels.shape)

***CNN Training***

from sklearn.datasets import fetch_openml

import numpy as np
import tensorflow as tf
from tensorflow import keras



import matplotlib.pyplot as plt
from datetime import datetime

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from random import randint

rnd_number = randint(0, len(loaded_labels))
plt.imshow(loaded_images[rnd_number])
plt.title(get_classlabel(loaded_labels[rnd_number]))

X_train, X_test, y_train, y_test = train_test_split(loaded_images, loaded_labels, test_size=1/7, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1/8, random_state=42)

print(X_train.shape, X_test.shape, X_val.shape)
print(y_train.shape, y_test.shape, y_val.shape)

np.save('/content/gdrive/MyDrive/animal_dataset_numpy/X_test_data.npy', X_test)
np.save('/content/gdrive/MyDrive/animal_dataset_numpy/y_test_data.npy', y_test)


from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from keras.layers import Activation, Dense

model = keras.models.Sequential()

model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(128,128,3), padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())

model.add(Dense(6, activation='softmax'))

model.summary()

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam

#sgd = SGD(learning_rate = 0.001)

model.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

X_train = X_train.astype('float32') / 255.0
y_train = y_train.astype('uint')
X_val = X_val.astype('float32') / 255.0
y_val = y_val.astype('uint')

model.fit(X_train, y_train, batch_size=60, epochs=8, validation_data=(X_val, y_val))

del model

import pandas as pd

pd.DataFrame(model.history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.show()

!mkdir -p saved_model
model.save('/content/gdrive/MyDrive/animal_dataset_numpy/RA_CNN')

***CNN Inference***

trained_model = tf.keras.models.load_model('/content/gdrive/MyDrive/animal_dataset_numpy/RA_CNN')

trained_model.summary()

loaded_X_test = np.load('/content/gdrive/MyDrive/animal_dataset_numpy/X_test_data.npy')
loaded_y_test = np.load('/content/gdrive/MyDrive/animal_dataset_numpy/y_test_data.npy')

# Inference
X_test = loaded_X_test.astype('float32') / 255.0
y_test = loaded_y_test.astype('uint')

loss, acc = trained_model.evaluate(X_test, y_test, verbose=1)

print('Accuracy: %.3f' % acc)


from sklearn.metrics import classification_report, confusion_matrix

y_proba = trained_model.predict(X_test)
y_pred = trained_model.predict_classes(X_test)

print('Confusion Matrix')
print(confusion_matrix(y_pred, y_test))

print('Classification Report')
print(classification_report(y_pred, y_test))

print(trained_model.trainable_variables)  # Trained Parameters

