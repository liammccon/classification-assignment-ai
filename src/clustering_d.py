''' Question D, Clustering
Devise a method to plot the decision boundaries for this dataset using the optimized
parameters. Explain your approach and plot your results. (10 marks)
'''

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
#from clustering_c import get_mean
from iris.subsets import *
from k_means import *


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
axis[0].set_title("K = 2")
axis[0].set_xlabel("Sepal Width")
axis[0].set_ylabel("Sepal length")

#Voronoi plugin requires a min of 4 points, so we will plot extra points far outside out range which will not affect the plot
out_of_view_high = [100000, 100000]
out_of_view_low = [-100000, -100000]

mean_points = [[sepal_width_mean_k2[0], sepal_length_mean_k2[0]], [sepal_width_mean_k2[1], sepal_length_mean_k2[1]], out_of_view_high, out_of_view_low]
#Compute Voronoi diagram
voronoi = Voronoi(mean_points)
figure = voronoi_plot_2d(voronoi, axis[0], show_points = False, line_colors='blue', line_width=2)

axis[0].set_xlim([1, 5])
axis[0].set_ylim([3, 9]) 

#Now for K = 3
meansK3, infoK3 = kMeansClustering(data, 3, 50)
sepal_width_mean_k3, sepal_length_mean_k3 = get_mean(infoK3, -1, SEPAL_WIDTH, SEPAL_LENGTH)

#TODO setosa_sepal_widths, virginica_sepal_widths, versicolor_sepal_widths
axis[1].plot(setosa_sepal_widths, setosa_sepal_lengths, 'o', color='blue')
axis[1].plot(virginica_sepal_widths, virginica_sepal_lengths, 'o', color='grey')
axis[1].plot(versicolor_sepal_widths, versicolor_sepal_lengths, 'o', color='green')
axis[1].plot(all_sepal_widths, all_sepal_lengths, 'o', color='black')
axis[1].plot(sepal_width_mean_k3, sepal_length_mean_k3, 'r^')
axis[1].set_title("K = 3")
axis[1].set_xlabel("Sepal Width")
axis[1].set_ylabel("Sepal length")

#Voronoi plugin requires a min of 4 points, so we will plot extra points far outside out range which will not affect the plot
out_of_view_high = [100000, 100000]
out_of_view_low = [-100000, -100000]

mean_points_k3 = [[sepal_width_mean_k3[0], sepal_length_mean_k3[0]], [sepal_width_mean_k3[1], sepal_length_mean_k3[1]], [sepal_width_mean_k3[2], sepal_length_mean_k3[2]], out_of_view_low]
#Compute Voronoi diagram
voronoi = Voronoi(mean_points_k3)
figure = voronoi_plot_2d(voronoi, axis[1], show_points = False, line_colors='blue', line_width=2)

axis[1].set_xlim([1, 5])
axis[1].set_ylim([3, 9]) 

plt.show()
