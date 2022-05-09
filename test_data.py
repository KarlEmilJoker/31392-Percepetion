import sklearn as sk
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import os
from tqdm import tqdm
import random
import numpy as np
import pickle


dataset_path = "/home/karl/DTU/Perception/dataset/test"


CATEGORIES = ["book", "box", "cup"]

for category in CATEGORIES:
    path = os.path.join(dataset_path, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
        plt.imshow(img_array, cmap='gray')  # graph it
        plt.show()  # display!
        break

IMG_SIZE = 100
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
print(new_array.shape)

training_data = []

def create_training_data():
    for category in CATEGORIES:

        path = os.path.join(dataset_path,category)  # create path to categories
        class_num = CATEGORIES.index(category)  # get the classification  (0, 1, or 2). 0=book 1=cardboardbox 2=cup
        k = 0
        for img in tqdm(os.listdir(path)):  # iterate over each image
            k = k+1

            if k <= 8:
                try:
                    img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                    training_data.append([new_array, class_num])  # add this to our training_data
                except Exception as e:  # in the interest in keeping the output clean...
                    pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

create_training_data()
#random.shuffle(training_data)

X = []
Y = []

for features, label in training_data:
    X.append(features)
    Y.append(label)

X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = np.array(Y)
pickle_out = open("X_test.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("Y_test.pickle","wb")
pickle.dump(Y, pickle_out)
pickle_out.close()
