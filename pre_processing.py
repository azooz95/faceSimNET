import os
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2 as cv

def convertor(path) -> str:
    return '/'.join(path.split('\\'))

def data(path, number_of_folder=1):
    data = []
    for i in os.listdir(path)[:number_of_folder]:
        print(f"Folder Name: {i}")
        n_path = os.path.join(path,i)
        for j in os.listdir(n_path):
            data.append(cv.imread(os.path.join(n_path,j)))
    return np.array(data)/255.0

def vis(data, no=25):
    row_col = int(np.sqrt(no))
    plt.figure(figsize=(15,12))
    r = np.random.randint(data.shape[0], size=no)
    for index,i in enumerate(r):
        plt.subplot(row_col,row_col, index+1)
        plt.imshow(data[i])
        plt.axis('off')
    plt.show()

def pair_data(data,positive=0.5):
    pr_data = []
    labels = []
    a = hot_encoding([1,0])
    for index,i in enumerate(data): 
        while True:
            r = random.randint(0, data.shape[0]-1)
            if index !=r:
                break
        
        pr_data += [[i,i]]
        pr_data += [[i,data[r]]]
        labels += [a[0],a[1]]

    return np.array(pr_data),np.array(labels)

def hot_encoding(list_of_labels):
    return np.eye(len(list_of_labels))

print(hot_encoding([1,2]))