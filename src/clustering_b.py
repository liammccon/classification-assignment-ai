''' Question B, Clustering
Plot the value of the objective function, Distortion, as a function of the iteration
to show that your learning rule and implementation minimizes this expression. (5
marks)
'''
from iris.subsets import *
from clustering import distortion as d
from k_means import *
import numpy as np
import matplotlib.pyplot as plt
import numpy as np


data = iris['data']
means, info = kMeansClustering(data, 3, 50)

print("Distortion for all iterations, running K-Means with K=3 and a max of 50 iterations")
for i in range(len(info['distortion'])):
    print(f'   Iteration: {i}, Distortion: {info["distortion"][i]}')

plt.plot(info['distortion'], "b-")
plt.xlabel('Iteration')
plt.ylabel('Distortion')
plt.title("Distortion for each iteration, k = 3")
plt.show()