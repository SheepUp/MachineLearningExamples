#!/usr/bin/python
#
# author:
#
# date:
# description:
# cat and dog images from kaggle - put all in a data folder separated into one 
# folder per class
#
from scipy.misc import imread, imshow, imresize
import os
import numpy as np

##
# General Machine Learning Models
##
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
# Dimensionality Reduction
from sklearn.decomposition import TruncatedSVD
# Ensemble Learning
from sklearn.ensemble import RandomForestClassifier


##
# Neural Networks (tensorflow+keras)
##
from keras.layers import Dense, Activation
from keras.layers import Dropout, Flatten
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D

# Procedures
def loadDataset(samples=100, flat=True):
    X = []
    y = []
    # load cats
    cats_dir = "data/cats"
    dogs_dir = "data/dogs"
    # cat images
    for cat_file in os.listdir(cats_dir)[:samples]:
        cat_img = imread(os.path.join(cats_dir, cat_file),flatten=True)
        if flat:
            cat_img = imresize(cat_img, size=(128,128)).flatten()
        else:
            cat_img = imresize(cat_img, size=(128,128))
        X.append(cat_img.astype("uint8"))
        y.append(1)
    # dog images
    for dog_file in os.listdir(dogs_dir)[:samples]:
        dog_img = imread(os.path.join(dogs_dir, dog_file),flatten=True)
        if flat:
            dog_img = imresize(dog_img, size=(128,128)).flatten()
        else:
            dog_img = imresize(dog_img, size=(128,128))
        X.append(dog_img.astype("uint8"))
        y.append(0)
    
    # Transform data into one big array
    X = np.stack(X)
    y = np.array(y)
    # normalize the input data
    X  = X.astype('float32')
    X /= 255

    return np.stack(X), np.array(y)

def splitDataset(X,y):
    X_num = X.shape[0]
    indicies = np.random.permutation(X_num)
    split_idx = int(X_num * 0.8)
    X_data, y_data = X[indicies[:split_idx]], y[indicies[:split_idx]]
    X_test, y_test = X[indicies[split_idx:]], y[indicies[split_idx:]]

    return X_data, y_data, X_test, y_test

def transformData(X_tr,X_test,C):
    svd  = TruncatedSVD(n_components=C)
    X_tr = svd.fit_transform(X_tr)
    X_test = svd.transform(X_test)
    return X_tr, X_test


def mlpModel():

    model = Sequential()
    model.add(Dense(128, input_dim=16384))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(Dropout(0.5)) 
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    
    return model


def cnnModel():

    model = Sequential()
    
    model.add(Convolution2D(32,3,3,border_mode='valid',input_shape=(128,128,1)))
    model.add(Activation("relu"))
    model.add(Convolution2D(32,3,3,border_mode='valid'))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(32,3,3,border_mode='valid'))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1,activation="sigmoid"))
    
    return model


# Main
def main_sklearn():
    X, y = loadDataset(samples=100)
    X_tr, y_tr, X_test, y_test = splitDataset(X,y)
    X_tr, X_test = transformData(X_tr,X_test,100)

    models = [
        ("KNN", KNeighborsClassifier()),
        ("DT", DecisionTreeClassifier()),
        ("LR", LogisticRegression()),
        ("SVC", SVC()),
        ("RF", RandomForestClassifier()),
    ]

    for (name,clf) in models:
        clf.fit(X_tr,y_tr)
        score = clf.score(X_test,y_test)
        print name, ":", score

def main_keras():
    X, y = loadDataset(samples=100, flat=True)
    X_tr, y_tr, X_test, y_test = splitDataset(X,y)
    
    # uncomment if conv net
    #X_tr = np.expand_dims(X_tr,3)
    #X_test = np.expand_dims(X_test,3)
    
    model = mlpModel()
    
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy','precision','recall','fscore'])

    model.fit(X_tr,y_tr, nb_epoch=20, verbose=1, batch_size=64, shuffle=True,
              validation_data=(X_test,y_test))
    
    score = model.evaluate(X_test,y_test)
    
    print "nn:", score
    
    

if __name__ == "__main__":
    #main_keras()
    pass
