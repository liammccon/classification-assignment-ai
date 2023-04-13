''' Question D, Clustering
Devise a method to plot the decision boundaries for this dataset using the optimized
parameters. Explain your approach and plot your results. (10 marks)
'''

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from clustering_c import get_mean
from iris.subsets import *
from k_means import kMeansClustering



points = [ [5.01224485, 3.43265302], [5.68071895, 2.66303558], [6.58400215, 2.99035867], [1.4591837 , 0.24693877], [4.09984023, 1.26282014]]


data = iris['data']
all_sepal_widths = np.concatenate((setosa_sepal_widths, virginica_sepal_widths, versicolor_sepal_widths))
all_sepal_lengths = np.concatenate((setosa_sepal_lengths, virginica_sepal_lengths, versicolor_sepal_lengths))

#Run kMeansClustering for K=2, with a max of 50 iterations
meansK2, infoK2 = kMeansClustering(data, 2, 50)
sepal_width_mean_k2, sepal_length_mean_k2 = get_mean(infoK2, -1, SEPAL_WIDTH, SEPAL_LENGTH)

figure, axis = plt.subplots(1, 2)
figure.suptitle("Voronoi boundaries", fontsize=16)
axis[0].plot(all_sepal_widths, all_sepal_lengths, 'o', color='black')
axis[0].plot(sepal_width_mean_k2, sepal_length_mean_k2, 'r^')

out_of_view_high = [100000, 100000]
out_of_view_low = [-100000, -100000]

mean_points = [[sepal_width_mean_k2[0], sepal_length_mean_k2[0]], [sepal_width_mean_k2[1], sepal_length_mean_k2[1]], out_of_view_high, out_of_view_low]
#Compute Voronoi diagram
voronoi = Voronoi(mean_points)
figure = voronoi_plot_2d(voronoi)

axis[0].set_xlim(1, 5)
axis[0].set_ylim(3, 9) 
plt.show()

#plot_voronoi(points)