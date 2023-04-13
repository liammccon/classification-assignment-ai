import math
from iris.subsets import *
from clustering.distortion import *
import numpy as np
import random
import copy

def kMeansClustering(data, k, iterations):
    """
    This function performs k-means clustering on the given data set.

    Parameters:
    data (list): The data set to cluster. Should be 2 dimensional, N x I, where N is the number of samples, and I is the number of features per sample.
    k (int): The number of clusters to create.
    iterations (int): The maximum number of iterations to run the algorithm for.

    Returns:
    means (list) : A list of k means that represent the final centers of each cluster.
    info (dict) : A dictionary that contains information about the algorithm's run for each iteration, such as distortion (the sum of squared distances between each point and its assigned mean) and means.
    """
    means = getKRandMeansFromData(data, k)
    
    #To store inner details about the algorithm's run
    info = {'distortion' : [],
            'means' : [] }
    
    info['distortion'].append(distortion(data, means))
    info['means'].append(copy.deepcopy(means))

    previousDistortion = distortion(data, means)

    for iteration in range(1, iterations + 1):
        samplesByCluster = [ []*1 for i in range(k)] #Create a 3D array with the following dimensions: [K clusters][N samples][I features]
        
        #add the sample to the closest cluster in samplesByCluster
        for sample in data:
            closestMean = closestMeanIndex(sample, means)
            samplesByCluster[closestMean].append(sample.copy())
        
        #Calculating the new mean values
        for cluster in range(k):
            features = len(data[0])
            for feature in range(features):
                sum = 0
                samples = len(samplesByCluster[cluster])
                for sample in range(samples):
                    sum += samplesByCluster[cluster][sample][feature]
                numOfSamplesInCluster = len(samplesByCluster[cluster])
                if(numOfSamplesInCluster == 0): 
                    #Avoid /0 errors in the situation where no samples were assigned to a cluster
                    numOfSamplesInCluster = 1
                means[cluster][feature] = sum / numOfSamplesInCluster #The learning rule for k-means

        newDistortion = distortion(data, means)
        converged = newDistortion == previousDistortion
        if converged:
            break
        else: 
            previousDistortion = newDistortion
        info['distortion'].append(distortion(data, means))
        info['means'].append(copy.deepcopy(means))

    return means, info

def get_mean(info, iteration, feature1, feature2):
    '''
    Get all of the means for two features of a specified iteration, to display on a 2D graph.
    Param
    iteration: the iteration you'd like to inspect, or -1 for the final iteration 
    Returns
    An array of all of the means for feature1 of the specified iteration
    An array of all of the means for feature2 of the specified iteration
    '''
    feature1Means = []
    feature2Means = []
    meansByIteration = info['means']
    if iteration == -1:
        iteration = len(meansByIteration) - 1
    for mean in range(len(meansByIteration[0])):
        feature1Means.append(meansByIteration[iteration][mean][feature1])
        feature2Means.append(meansByIteration[iteration][mean][feature2])
    return feature1Means, feature2Means

def removeSkippedClusters(means, clustersToSkip):
    newMeans = []
    for mean in range(len(means)):
        if mean not in clustersToSkip:
            newMeans.append(means[mean])
    return newMeans

#Returns an array of K random means within the range of values in data
def getKRandomMeans(data: list, k):
    rotatedData = np.rot90(data.copy()) #Returns an array with I features as the rows and the N samples as the columns
    minOfEachFeature = []
    maxOfEachFeature = []
    means = []
    for feature in range(len(rotatedData)):
        minOfEachFeature.append(min(rotatedData[feature]))
        maxOfEachFeature.append(max(rotatedData[feature]))
    
    for i in range(k):
        mean = []
        for feature in range(len(rotatedData)):
            mean.append(random.uniform(minOfEachFeature[feature], maxOfEachFeature[feature]))
        means.append(mean)
    
    return means

#Selects k unique samples at randomo from the data to be the means
def getKRandMeansFromData(data: list, k):
    samplesIndexes = []
    while len(samplesIndexes) < k:
        randIndex = random.randint(0, len(data) - 1)
        if randIndex not in samplesIndexes:
            samplesIndexes.append(randIndex)
    means = []
    for sample in samplesIndexes:
        means.append(data[sample])
    return means

def getRandomMean(data):
    return getKRandomMeans(data, 1)[0]