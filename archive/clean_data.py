#!/usr/bin/env python3

import os, random, shutil

def main():
    path = "./data/data/validation"
    for dirName, subdirList, fileList in os.walk(path, topdown=False):
        if path == dirName:
            continue
        author = dirName.split("/")[-1]
        train_size = len(fileList)
        print(author, train_size)

        #val_files = random.sample(fileList, val_size)
        #for f in val_files:
        #    shutil.move(path+"/"+author+"/"+f, "./data/data/validation/"+author)


if __name__ == '__main__':
    main()