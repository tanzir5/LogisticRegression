import math
import random
import numpy as np
import sys

INF = 1000000000
class Dataset:
    dataFeature = None
    dataResult = np.array([])
    count = int()
    nameToColumn = {"age": 0, "edu-num" : 4, "capital-gain" : 10, "capital-loss" : 11, "hours-per-week" : 12}
    def __init__(self, s):
        file = open(s,"r")
        f = file.readlines()
        row = len(f)-1
        col = len(self.nameToColumn)+1
        self.dataFeature = np.zeros(shape=(row, col))
        self.count = 0
        total = 0
        for line in f:
            total += 1
            tmp = [1]
            inputs = list(map(str, line.split(", ")))
            if(len(inputs) == 1):
                break
            for feature in self.nameToColumn:
                tmp.append(float(inputs[self.nameToColumn[feature]]))
            self.dataFeature[self.count] = tmp
            self.count+=1
            if (inputs[-1].startswith("<")):
                self.dataResult = np.append(self.dataResult, 0)
            else:
                self.dataResult = np.append(self.dataResult, 1)
        
        print(self.dataResult)
        print(self.count)
        for f in range(1,6):
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

def totalCost(my, theta):
    ret = 0
    for i in range(0,my.count):
        ret += cost(h(theta,my.dataFeature[i]), my.dataResult[i])
    ret /= my.count
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

alpha = 1.5
iteration = 2000
name = "data/adult.data"
testName = "data/adult.test"
print("training set initialization")
my = Dataset(name)
print("end")

print("test set initialization")
test = Dataset(testName)
print("end")
theta = []
totalFeature = 6
for i in range(0,totalFeature):
    theta.append(random.uniform(-1.0,2))

for i in range(0, iteration):
    c = totalCost(my, theta)
    print(accuracy(test, theta))
    print("Average cost in " + str(i) + "-th iteration: " + str(c))
    costList = []
    for j in range(0, my.count):
        costList.append(h(theta, my.dataFeature[j]) - my.dataResult[j])
    npCost = np.array(costList)
    tmp = np.dot(npCost, my.dataFeature)
    tmp /= my.count
    tmp *= alpha
    theta = np.subtract(theta, tmp)
    print(theta)

