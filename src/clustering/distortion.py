import math
#N samples, each with I features
#K Clusters
#K Means, each with I features

#Returns the "Distortion" between each sample and its mean
#means should be a K x I array of all the current means K clusters, each with I features
#Data should be an N x I dataset
def distortion(data: list, means: list) -> int:
    distortion = 0
    k = len(means)
    for sample in data:
        closestMean = means[closestMeanIndex(sample, means)]
        distortion += distance(sample, closestMean)
    return distortion


#Returns the index of the mean the sample is closest to.
def closestMeanIndex(sample, means) -> int:
    bestDistance = math.inf
    closestMean = -1
    for i in range(len(means)):
        newDistance = distance(sample, means[i])
        if newDistance < bestDistance:
            closestMean = i
            bestDistance = newDistance
    return closestMean

#Returns the distance between the sample and the mean
#Calculated as the squared sum of the difference between each feature of the sample and mean
def distance(sample, mean):
    if( len(sample) != len(mean)):
        raise Exception("sample and mean must contain the same number of features")
    distance = 0
    for feature in range(len(sample)):
        distance += (sample[feature] - mean[feature]) ** 2
    return distance