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
        print(row)
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
        
        
        print(self.dataResult)
        print(self.count)
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

alpha = 0.01
iteration = 20000
name = "marks.data"
print("training set initialization")
my = Dataset(name, [0,1])
print("end")

theta = []
totalFeature = 3
for i in range(0,totalFeature):
    theta.append(random.uniform(-5.0,5))

for i in range(0, iteration):
    c = totalCost(my, theta)
    print(accuracy(my, theta))
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


