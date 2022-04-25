#!/usr/bin/env python3

'''
used to clean and process kaggle data
'''

import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import imageio
import cv2
from skimage.filters import prewitt_h,prewitt_v
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import skimage

class Calligraphers:
    def __init__(self, names):
        self.num_to_name = set(names)    
        self.name_to_num = {name:num for num, name in enumerate(self.num_to_name)}
        self.enc = OneHotEncoder().fit([names])

    def len(self):
        return len(self.num_to_name)

    def names(self):
        return self.num_to_name

    # turn list of names into list of numbers
    def onehotencode_list(self, list):
        num_list = self.numberize_list(list)
        list = [[x] for x in num_list]
        return self.enc.fit_transform(list).toarray()

    # turn list of names into onehotencoding list
    def numberize_list(self, list):
        num_list = []
        for name in list:
            num_list.append(self.numberize(name))
        return np.array(num_list)

    def numberize(self, name):
        if name in self.name_to_num:
            return self.name_to_num[name]

    def denumberize(self, num):
        return self.num_to_name[num]

# returns path_x = np array of image matrices of size (64*64,) (i.e. path_x[0] = image #0 pixel matrix)
# returns path_y = np array of calligraphers (i.e. path_y[0] = lqs)
# save both as .npy files
def save_data(pathname):
    path_y = []
    path_x = []
    path = "archive/data/data/" + pathname
    for dirName, subdirList, fileList in os.walk(path, topdown=False):
        if dirName == path:
            continue
        y = dirName.split(path+'/')[1]
        x = []
        for f in fileList:
            filename = path + '/' + y + '/' + f
            im = imageio.imread(filename, as_gray=True)
            im = hog(im)
            path_x.append(im)
            path_y.append(y)

    path_x = np.array(path_x)
    path_y = np.array(path_y)

    print(f"Shape of training set (# of samples, width, height): {path_x.shape}")
    print(f"Shape of testing set (# of samples, width, height): {path_y.shape}")

    # reshape x
    path_x = path_x.reshape(-1, 64*64)
    path_x = path_x.astype('float32')
    path_x /= 255
    np.save('hog/'+pathname+'/x.npy', path_x)
    np.save('hog/'+pathname+'/y.npy', path_y)
    return path_x, path_y

# load any image and display it
def show_image(image):
    image = image.reshape(64,64)
    plt.imshow(image, cmap='Greys')
    plt.show()

def hog(image):
    image = image.reshape(64,64)
    fd, image = skimage.feature.hog(image, visualize=True)
    return image

# create numpy arrays of data and save
def create_numpy_files():
    x_train, y_train = save_data("train")
    x_test, y_test = save_data("test")
    x_valid, y_valid = save_data("validation")

# load numpy arrays of already nice data -- must run create_numpy_files() first
def load_data():
    x_train = np.load('data/train/x.npy', allow_pickle=True)
    y_train = np.load('data/train/y.npy', allow_pickle=True)
    x_test = np.load('data/test/x.npy', allow_pickle=True)
    y_test = np.load('data/test/y.npy', allow_pickle=True)
    x_valid = np.load('data/validation/x.npy', allow_pickle=True)
    y_valid = np.load('data/validation/y.npy', allow_pickle=True)

def main():
    save_data("test")
    '''
    x_train = np.load('data/train/x.npy', allow_pickle=True)
    y_train = np.load('data/train/y.npy', allow_pickle=True)
    x_valid = np.load('data/validation/x.npy', allow_pickle=True)
    y_valid = np.load('data/validation/y.npy', allow_pickle=True)
    hog(x_train[1])

    x_train_hu = []
    for x in x_train:
        f = hog(x)
        x_train_hu.append(f)

    x_valid_hu = []
    for x in x_valid:
        f = hog(x)
        x_valid_hu.append(f)

    k = 1
    neigh = KNeighborsClassifier(n_neighbors=k, weights='distance', n_jobs=-1, p=2)
    neigh.fit(x_train_hu, y_train)
    y_pred = neigh.predict(x_valid_hu)
    print(f'kneighbor k={k} accuracy={accuracy_score(y_valid, y_pred)}')
    '''

if __name__ == '__main__':
    main()