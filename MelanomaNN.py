import pandas as pd
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

np.random.seed(42)
##from keras.utils.np_utils import to_categorical 
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

folder_malignant_train = 'folder/skin-cancer-malignant-vs-benign/train/malignant'

folder_benign_test = 'skin-cancer-malignant-vs-benign/test/benign'
folder_malignant_test = 'skin-cancer-malignant-vs-benign/test/malignant'

read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))

# training pictures 
ims_benign = []
ims_malignant = []

for filename in os.listdir(folder_benign_train):
    img = Image.open(os.path.join(folder_benign_train, filename))
    img = img.convert("RGB")  # Ensure that the image is in RGB format
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
