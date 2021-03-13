import numpy as np
import pandas as pd
import math
from math import sqrt
import time
import random

#Use your own folder when testing
iris_df = pd.read_csv('D:/HWData/iris.csv')
    
def getEucilidDistance(flower1, flower2):
    if flower1 == None or flower2 == None:
        return float("inf")
    result = 0
    for i in range(len(flower1)-1):
        result += (flower1[i]-flower2[i])**2;
    result = sqrt(result);
    return result

def getCosineDistance(flower1, flower2):
    if flower1 == None or flower2 == None:
        return 0
    result = 0
    flower1SquareSum = 0
    flower2SquareSum = 0
    for i in range(len(flower1)-1):
        result += flower1[i]*flower2[i]
        flower1SquareSum += flower1[i]**2
        flower2SquareSum += flower2[i]**2
    return result/sqrt(flower1SquareSum*flower2SquareSum)

def getJaccardDistance(flower1, flower2):
    if flower1 == None or flower2 == None:
        return 1
    minSum = 0
    maxSum = 0
    for i in range(len(flower1)-1):
        if(flower1[i]<flower2[i]):
            minSum += flower1[i]
            maxSum += flower2[i]
        else:
            minSum += flower2[i]
            maxSum += flower1[i]
    return 1 - (minSum/maxSum)

def assign(instance, centroids, distance_func):
    minDistance = distance_func(instance, centroids[0])
    minDistanceIndex = 0
    for i in range(1, len(centroids)):
        d = distance_func(instance, centroids[i])
        if(distance_func == getCosineDistance and d > minDistance):
            minDistance = d
            minDistanceIndex = i
        elif(distance_func != getCosineDistance and d < minDistance):
            minDistance = d
            minDistanceIndex = i
    return minDistanceIndex
def createEmptyListOfLists(numSubLists):
    result=[]
    for i in range(numSubLists):
        result.append([])
    return result

def assignAll(instances, centroids, distance_func):
    clusters = createEmptyListOfLists(len(centroids))
    for instance in instances:
        clusterIndex = assign(instance, centroids, distance_func)
        clusters[clusterIndex].append(instance)
    return clusters

def getNewCentroids(clusters):
    centroids=[]
    for i in range(len(clusters)):
        centroid = getNewCentroid(clusters[i])
        centroids.append(centroid)
    return centroids
def getNewCentroid(cluster):
    if (len(cluster) == 0):
        return
    num = len(cluster[0])
    result = [0]*num
    for instance in cluster:
        for i in range(0,num-1):
            result[i]+=instance[i]
    for i in range(0,num-1):
        result[i]/= float(len(cluster))
    return tuple(result)

def computeWithinss(clusters, centroids, distance_func):
    result = 0
    for i in range(len(centroids)):
        centroid = centroids[i]
        cluster = clusters[i]
        for instance in cluster:
            result += distance_func(centroid, instance)
    return result
def getAccuracy(clusters):
    result = [0, 0, 0]
    total_correct = 0
    total = 0
    for i in range(len(clusters)):
        result = [0,0,0]
        for j in range(len(clusters[i])):
            if clusters[i][j][4] == 'Iris-setosa':
                result[0] +=1
            elif clusters[i][j][4] == 'Iris-versicolor':
                result[1] +=1
            elif clusters[i][j][4] == 'Iris-virginica':
                result[2] +=1
        currmax = result[0]
        if(result[1]>currmax):
            currmax = result[1]
        if(result[2]>currmax):
            currmax = result[2]
        total += len(clusters[i])
        total_correct += currmax
    return total_correct/total

def kmeans(instances, k, distance_func, termination, iteration_limit):
    start = time.time()
    result={}
    random.seed(time.time())
    #get random starting centroids
    centroids = random.sample(instances, k)
    prevCentroids=[]
    prevSSE = -2
    currSSE = -1
    iter = 0
    if(termination == 'no_change'):
        currState = centroids
        prevState = prevCentroids
    elif(termination == 'SSE_increse'):
        currState = currSSE
        prevState = prevSSE
    elif(termination == 'preset_value'):
        currState = iter
        prevState = iteration_limit
    else:
        currState = centroids
        prevState = prevCentroids
    while(prevState != currState):
        #change iteration to compare with preset_value
        iter+=1
        clusters=assignAll(instances, centroids, distance_func)
        prevCentroids = centroids
        centroids = getNewCentroids(clusters)
        prevSSE = currSSE
        currSSE = computeWithinss(clusters, centroids, distance_func)
        if(termination == 'no_change'):
            currState = centroids
            prevState = prevCentroids
        elif(termination == 'SSE_increse'):
            if(prevSSE != -1):
                if(currSSE >= prevSSE and distance_func!=getCosineDistance):
                    break
                elif(currSSE <= prevSSE and distance_func==getCosineDistance):
                    break
        elif(termination == 'preset_value'):
            currState = iter
            prevState = iteration_limit
        else:
            currState = centroids
            prevState = prevCentroids
    end = time.time()
    result["clusters"] = clusters
    result["centroids"] = centroids
    result["sse"] = currSSE
    result["accuracy"] = getAccuracy(clusters)
    result["time"]=end-start
    result["iter"]=iter
    return result
#Q1################################################################################
print('Q1')
clustering1 = kmeans(iris_df.values.tolist(), 3, getEucilidDistance, 'no_change', 0)
clustering2 = kmeans(iris_df.values.tolist(), 3, getCosineDistance, 'no_change', 0)
clustering3 = kmeans(iris_df.values.tolist(), 3, getJaccardDistance, 'no_change', 0)
print('Euc_sse',clustering1["sse"])
print('Cos_sse',clustering2["sse"])
print('Jar_sse',clustering3["sse"])
# #Q2################################################################################
print('Q2')
print('Euc_accuracy',clustering1["accuracy"])
print('Cos_accuracy',clustering2["accuracy"])
print('Jar_accuracy',clustering3["accuracy"])
# #Q3################################################################################
print('Q3')
print('Euc_iter',clustering1["iter"])
print('Cos_iter',clustering2["iter"])
print('Jar_iter',clustering3["iter"])
 
print('Euc_time',clustering1["time"])
print('Cos_time',clustering2["time"])
print('Jar_time',clustering3["time"])
# #Q4################################################################################
print('Q4')
clustering1 = kmeans(iris_df.values.tolist(), 3, getEucilidDistance, 'SSE_increse', 0)
clustering2 = kmeans(iris_df.values.tolist(), 3, getCosineDistance, 'SSE_increse', 0)
clustering3 = kmeans(iris_df.values.tolist(), 3, getJaccardDistance, 'SSE_increse', 0)
 
print('Euc_iter',clustering1["iter"])
print('Cos_iter',clustering2["iter"])
print('Jar_iter',clustering3["iter"])
 
print('Euc_time',clustering1["time"])
print('Cos_time',clustering2["time"])
print('Jar_time',clustering3["time"])
 
clustering1 = kmeans(iris_df.values.tolist(), 3, getEucilidDistance, 'preset_value', 100)
clustering2 = kmeans(iris_df.values.tolist(), 3, getCosineDistance, 'preset_value', 100)
clustering3 = kmeans(iris_df.values.tolist(), 3, getJaccardDistance, 'preset_value', 100)
 
print('Euc_time',clustering1["time"])
print('Cos_time',clustering2["time"])
print('Jar_time',clustering3["time"])




