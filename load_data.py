#!/usr/bin/env python3

import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

class Calligraphers:
    def __init__(self, names):
        self.num_to_name = list(names)    
        self.name_to_num = {name:num for num, name in enumerate(self.num_to_name)}
        self.enc = OneHotEncoder().fit([names])

    def len(self):
        return len(self.num_to_name)

    def names(self):
        return self.num_to_name

    # turn list of names into list of numbers
    def onhotencode_list(self, list):
        num_list = self.numberize_list(list)
        list = [[x] for x in num_list]
        return self.enc.fit_transform(list).toarray()

    # turn list of names into onehotencoding list
    def numberize_list(self, list):
        num_list = []
        for name in list:
            num_list.append(self.numberize(name))
        return num_list

    def numberize(self, name):
        if name in self.name_to_num:
            return self.name_to_num[name]

    def denumberize(self, num):
        return self.num_to_name[num]

# returns path_x = list of list of image paths (i.e. path_x[0][10] = pathname of lqs image #10)
# returns path_y = list of calligraphers (i.e. path_y[0] = lqs)
def load_data(pathname):
    path_y = []
    path_x = []
    path = "archive/data/data/" + pathname
    for dirName, subdirList, fileList in os.walk(path, topdown=False):
        if dirName == path:
            continue
        y = dirName.split(path+'/')[1]
        path_y.append(y)
        x = []
        for f in fileList:
            filename = path + '/' + y + '/' + f
            x.append(filename)
        path_x.append(x)
    return path_x, path_y

def show_image(imagepath):
    im = Image.open(imagepath)
    plt.imshow(im, cmap='Greys')
    plt.show()

def main():
    train_x, train_y = load_data("train")
    test_x, test_y = load_data("test")
    valid_x, valid_y = load_data("validation")

    calligraphers = Calligraphers(train_y)

if __name__ == '__main__':
    main()
