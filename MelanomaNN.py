import pandas as pd
import time
start_time = time.time()
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OrdinalEncoder
from keras.applications import ResNet50
from sklearn.metrics import accuracy_score 
from keras.layers import Input, Flatten, Dense
from keras.models import Model

from keras import layers
from keras import models

num_classes = 10
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)  
predictions = Dense(num_classes, activation='softmax')(x)  

model = Model(inputs=model.input, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


np.random.seed(42)
# from keras.utils.np_utils import to_categorical 
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import LabelEncoder
# import autokeras as ak

import keras
##from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import LabelEncoder

folder_benign_train = 'skin-cancer-malignant-vs-benign/train/benign'
folder_malignant_train = 'skin-cancer-malignant-vs-benign/train/malignant'

folder_benign_test = 'skin-cancer-malignant-vs-benign/test/benign'
folder_malignant_test = 'skin-cancer-malignant-vs-benign/test/malignant'

read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))

# training pictures 
ims_benign = []
ims_malignant = []

for filename in os.listdir(folder_benign_train):
    img = Image.open(os.path.join(folder_benign_train, filename))
    img = img.convert("RGB") 
    ims_benign.append(np.array(img, dtype='uint8'))

X_benign = np.array(ims_benign)

for filename in os.listdir(folder_malignant_train):
    img = Image.open(os.path.join(folder_malignant_train, filename))
    img = img.convert("RGB")  # Ensure that the image is in RGB format
    ims_malignant.append(np.array(img, dtype='uint8'))

X_malignant = np.array(ims_malignant)

# testing pictures
ims_benign = [read(os.path.join(folder_benign_test, filename)) for filename in os.listdir(folder_benign_test)]
X_benign_test = np.array(ims_benign, dtype='uint8')
ims_malignant = [read(os.path.join(folder_malignant_test, filename)) for filename in os.listdir(folder_malignant_test)]
X_malignant_test = np.array(ims_malignant, dtype='uint8')

# labels
Y_benign = np.zeros(X_benign.shape[0])
Y_malignant = np.ones(X_malignant.shape[0])

Y_benign_test = np.zeros(X_benign_test.shape[0])
Y_malignant_test = np.ones(X_malignant_test.shape[0])


# merge 
X_train = np.concatenate((X_benign, X_malignant), axis = 0)
Y_train = np.concatenate((Y_benign, Y_malignant), axis = 0)

X_test = np.concatenate((X_benign_test, X_malignant_test), axis = 0)
Y_test = np.concatenate((Y_benign_test, Y_malignant_test), axis = 0)

# shuffling
s = np.arange(X_train.shape[0])
np.random.shuffle(s)
X_train = X_train[s]
Y_train = Y_train[s]

s = np.arange(X_test.shape[0])
np.random.shuffle(s)
X_test = X_test[s]
Y_test = Y_test[s] 

w=40
h=30
fig=plt.figure(figsize=(12, 8))
columns = 5
rows = 3

for i in range(1, columns*rows +1):
    ax = fig.add_subplot(rows, columns, i)
    if Y_train[i] == 0:
        ax.title.set_text('Benign')
    else:
        ax.title.set_text('Malignant')
    plt.imshow(X_train[i], interpolation='nearest')
plt.show() 


benign_train_count = Y_train[np.where(Y_train == 0)].shape[0]
malignant_train_count = Y_train[np.where(Y_train == 1)].shape[0]

print("Training Data:")
print(f"Benign Count: {benign_train_count}")
print(f"Malignant Count: {malignant_train_count}")

# Test Data
benign_test_count = Y_test[np.where(Y_test == 0)].shape[0]
malignant_test_count = Y_test[np.where(Y_test == 1)].shape[0]

print("\nTest Data:")
print(f"Benign Count: {benign_test_count}")
print(f"Malignant Count: {malignant_test_count}")

X_train = X_train/155.
X_test = X_test/155.

from sklearn.svm import SVC


model.fit(X_train.reshape(X_train.shape[0],-1), Y_train)
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test.reshape(X_test.shape[0],-1))
print("\nScore:", accuracy_score(Y_test, y_pred))
print ("\nRun Time:", time.time() - start_time)