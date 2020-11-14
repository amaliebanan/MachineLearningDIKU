import numpy as np
from scipy.stats import bernoulli
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

def tryIt(file):
    array = file.reshape((28,28)).T
    plt.imshow(array,cmap='gray',interpolation=None)
    #plt.show()


def euclid_distance(v1,v2):
    return np.sqrt(np.dot((v1-v2).T,(v1-v2)))


def find_neighbors(train_set, targetpoint,label_set, k):
    distances = [(euclid_distance(train_set[i],targetpoint),train_set[i],label_set[i]) for i in range(len(train_set))]
    sorted_data = sorted(distances)
    k_neighbors = [sorted_data[i] for i in range(0,k)]
    return k_neighbors

def predict(train_set,targetpoint,label_set,k,digits):
    neighbors = find_neighbors(train_set,targetpoint,label_set,k)
    count = 0
    for i in range(len(neighbors)):
        if neighbors[i][2] == digits[0]:
            count += 1
    if count >= (len(neighbors)+1)/2:
        return digits[0]
    else: return digits[1]



def pick_correct_images(train_set, digits, label_set):
    '''
    :param train_set: The whole training set cropped
    :param digits: List of the digits we want to compare
    :param label_set: The label set
    :return: Two lists - one with the 28x28 images that equals one of the digits in digit list,
    and one with the corresponding label
    '''
    input,labels = [],[]
    for i in range(len(train_set)):
        if label_set[i] in digits:
            input.append(train_set[i])
            labels.append(label_set[i])
    return input, labels

digits = [5,6]
X,Y = pick_correct_images(datafile,digits,labelfile)# X = input, Y = labels
trainX, valX, trainY, valY = train_test_split(X,Y,test_size=0.2,random_state=42) #Split the dataset into training and val

def plot_error_rate(trainX,trainY,valX,valY,digits):
    list_to_plot = []
    for k in range(1,34,2):
        count = 0
        for i in range(len(valX)):
            if predict(trainX,valX[i],trainY,k,digits) != valY[i]: #Get all the misclassifications
                count += 1
        list_to_plot.append((k,count/len(valY)))#for k, the ratio of misclassifications compared to total length of 28x28 images
    plt.plot(*zip(*list_to_plot))

    plt.show()
plot_error_rate(trainX,trainY,valX,valY,digits)
#print(count,len(valX), count/len(valX))

##For test files##
testX,testY = pick_correct_images(testfile,digits,testlabel)
plot_error_rate(testX,testY,testX,testY,digits)

'''
list_to_plot = []
for k in range(1,34,2):
    count = 0
    for i in range(len(TestX)):
        if predict(TestX,TestX[i],TestY,k,digits) == TestY[i]:
            count += 1
    list_to_plot.append((k,1-(count/len(TestY))))
#print(list_to_plot)
#print(count,len(X1), count/len(Y1))
#plt.plot(*zip(*list_to_plot))
#plt.show()
'''
