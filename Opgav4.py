import numpy as np
from scipy.stats import bernoulli
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import scipy.spatial as scipy
import pandas as pd

############## Opgave 4, K-NN ##############
#Parser
def parse_datafile(fname):
    with open(fname) as f:
        raw = f.read().split()
        n = 28 ** 2
        data = np.array([raw[i:i + n] for i in range(0, len(raw), n)], dtype=float)
    return data

### Parser
def parse_labelfile(fname):
    with open(fname) as f:
        labels = np.array(f.read().split(), dtype=int)
    return labels

datafile = parse_datafile("MNIST-Train-cropped.txt")
labelfile = parse_labelfile("MNIST-Train-Labels-cropped.txt")
testfile = parse_datafile("MNIST-Test-cropped.txt")
testlabel = parse_labelfile("MNIST-Test-Labels-cropped.txt")

def pick_correct_images(train_set, digits, label_set):
    '''
    :param train_set: The whole training set cropped
    :param digits: List of the digits we want to compare
    :param label_set: The label set
    :return: Two lists - one with the 28x28 images that equals one of the digits in digit list,
    and one with the corresponding label
    '''
    tempLabels = []
    input,labels = [],[]
    for i in range(len(train_set)):
        if label_set[i] in digits:
            input.append(train_set[i])
            tempLabels.append(label_set[i])
    for i in tempLabels:
        if i == digits[0]:
            labels.append(-1)
        else: labels.append(1)
    return input, labels

digits = [5,6]
X,Y = pick_correct_images(datafile,digits,labelfile)# X = input, Y = labels
trainX, valX, trainY, valY = train_test_split(X,Y,test_size=0.2,random_state=42) #Split the dataset into training and val

def predict2(trainx,x,trainy,k):
    m = np.array([x,]*len(trainx))  #Find distances
    m1 = trainX - m
    U = np.dot(m1,m1.T)
    dist = np.diag(U)

    dist_m = np.argsort(dist)[:k] #Get the k-shortests distances by index (k-neighbors)
    k_nearest_labels = [trainy[dist_m[i]] for i in range(0,k)] #Get labels of the k-shortests distances (k-neighbors)

    ksum = np.cumsum(k_nearest_labels)[:k]
    my_predictions = [] #Array of len 33, holding my guess of the label for x for k=1,3,5,7...
    for i in range(1,34,2):
        if ksum[i-1] > 0:
            my_predictions.append(1)
        else: my_predictions.append(-1)

    return my_predictions

def plot_error_rate(trainx,valX,trainy,valY,k):
    M = []
    for i in range(len(valX)):
        prediction = predict2(trainx,valX[i],trainy,k)
        M.append(prediction)
    m = (np.dot(np.dot(valY,M),(-1))+len(valX))/len(valX) #1xn.nx33 = (n-#fejl). Som jeg så ganger (-1) og +n for at få fejl

    plt.plot(m)
    plt.show()
    return m

#plot_error_rate(trainX,valX,trainY,valY,33)

##For test files##
testX,testY = pick_correct_images(testfile,digits,testlabel)
print(len(testX),len(testY))
plot_error_rate(trainX,testX,trainY,testY,33)
