import math
import random
import numpy as np
import sys

INF = 1000000000
class Dataset:
    dataFeature = None
    dataResult = np.array([])
    count = 0
    def __init__(self, s, needColumn):
        file = open(s,"r")
        f = file.readlines()
        row = len(f)-1
        col = len(needColumn)+1
        self.dataFeature = np.zeros(shape=(row, col))
        self.count = 0
        total = 0
        for line in f:
            total += 1
            tmp = [1]
            inputs = list(map(str, line.split(",")))
            inputs[len(inputs)-1] = inputs[len(inputs)-1][:-1]

            if(len(inputs) == 1):
                break
            for feature in needColumn:
                tmp.append(float(inputs[feature]))
            self.dataFeature[self.count] = tmp
            self.count+=1
            self.dataResult = np.append(self.dataResult, float(inputs[-1]))
        
        for f in range(1,col):
            mn = INF
            mx = -INF
            for d in self.dataFeature:
                mn = min(mn, d[f])
                mx = max(mx, d[f])
            
            for i in range(0, self.count):
                self.dataFeature[i][f] = (self.dataFeature[i][f] - mn)/(mx-mn)


def eval(theta, x):
    return np.dot(theta,x)

def h(theta, x):
    ret = np.dot(theta,x)
    ret = 1.0/(1+math.exp(-ret))
    return ret

def cost(hVal, y):
    ret = -y*math.log(hVal) - (1-y) * math.log(1-hVal)
    return ret

def totalCost(trainingSet, theta):
    ret = 0
    for i in range(0,trainingSet.count):
        ret += cost(h(theta,trainingSet.dataFeature[i]), trainingSet.dataResult[i])
    ret /= trainingSet.count
    return ret

def accuracy(test, theta):
    ret = 0
    for i in range(0,test.count):
        if(h(theta,test.dataFeature[i]) > 0.5):
            prediction = 1
        else:
            prediction = 0
        if(prediction == test.dataResult[i]):
            ret+=1
    return float(ret)/test.count * 100

alpha = None
iteration = None
trainingSetName = None
testSetName = None
trainingSet = None
testSet = None
theta = None
needColumn = None

def loadParameters():
    global alpha, iteration, trainingSetName, testSetName, needColumn
    file = open("configure.txt","r")
    f = file.readlines()

    line = f[0]
    inputs = list(map(str, line.split(" ")))
    alpha = float(inputs[1])

    line = f[1]
    inputs = list(map(str, line.split(" ")))
    iteration = int(inputs[1])
    
    line = f[2]
    inputs = list(map(str, line.split(" ")))
    inputs[1] = inputs[1][:-1]
    trainingSetName = inputs[1]
    
    line = f[3]
    inputs = list(map(str, line.split(" ")))
    inputs[1] = inputs[1][:-1]
    testSetName = inputs[1]

    line = f[4]
    inputs = list(map(str, line.split(" ")))
    inputs[1] = inputs[1][:-1]
    needColumn = list(map(int, inputs[1].split(",")))


def init():
    global trainingSet, testSet, theta
    loadParameters()
    
    print("training set initialization started")
    trainingSet = Dataset(trainingSetName, needColumn)
    print("end\n")

    print("test set initialization started")
    testSet = Dataset(testSetName, needColumn)
    print("end\n")

    theta = []
    totalFeature = len(needColumn)+1
    for i in range(0,totalFeature):
        theta.append(random.uniform(-5.0,5))

def update():
    global trainingSet, theta
    costList = []
    for j in range(0, trainingSet.count):
        costList.append(h(theta, trainingSet.dataFeature[j]) - trainingSet.dataResult[j])
    npCost = np.array(costList)
    tmp = np.dot(npCost, trainingSet.dataFeature)
    tmp /= trainingSet.count
    tmp *= alpha
    theta = np.subtract(theta, tmp)

def main():
    global alpha, iteration, trainingSetName, testSetName, trainingSet, testSet, theta
    init()
    
    for i in range(0, iteration):
        c = totalCost(trainingSet, theta)
        print("Iteration number:",i+1, end = " ")
        print("Accuracy:",accuracy(trainingSet, theta), end = " ")
        print("Average cost:", c)
        update()


if __name__ == "__main__":
    main()
