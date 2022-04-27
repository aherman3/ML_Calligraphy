import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import imageio
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from joblib import dump, load
'''
load numpy files of data and shuffle for variety
'''

drive_path = "/afs/crc.nd.edu/group/dmsquare/vol5/yyan6/ML/HOG/"
x_train = np.load(drive_path+'train/x.npy', allow_pickle=True)
y_train = np.load(drive_path+'train/y.npy', allow_pickle=True)
x_train, y_train = shuffle(x_train, y_train)
#x_test = np.load(drive_path+'test/x.npy', allow_pickle=True)
#y_test = np.load(drive_path+'test/y.npy', allow_pickle=True)
#x_test, y_test = shuffle(x_test, y_test)
x_valid = np.load(drive_path+'validation/x.npy', allow_pickle=True)
y_valid = np.load(drive_path+'validation/y.npy', allow_pickle=True)
x_valid, y_valid = shuffle(x_valid, y_valid)


'''
SoftMax Classifier
'''

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(multi_class = 'multinomial',random_state=0, max_iter = 500).fit(x_train, y_train)
y_pred = clf.predict(x_valid)
print('accuracy=',str(accuracy_score(y_valid, y_pred)))
dump(clf, 'softmax_hog.joblib')
