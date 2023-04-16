''' Question C of Clustering
Plot the results of the learning process by showing the initial, intermediate, and con-
verged cluster centers overlaid on the data for k = 2 and k = 3. (5 marks)
'''
from iris.subsets import *
from k_means import *
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

data = iris['data']

#Run kMeansClustering for K=2, with a max of 50 iterations
meansK2, infoK2 = kMeansClustering(data, 2, 50)

#Plot the data for the petal width and length
all_petal_widths = np.concatenate((setosa_petal_widths, versicolor_petal_widths, virginica_petal_widths))
all_petal_lengths = np.concatenate((setosa_petal_lengths, versicolor_petal_lengths, virginica_petal_lengths))

#Get the starting, intermediate, and converged means for k = 2
starting_petal_width_K2, starting_petal_length_K2 = get_mean(infoK2, 0, PETAL_WIDTH, PETAL_LENGTH)
mid_petal_width_K2, mid_petal_length_K2 = get_mean(infoK2, 1, PETAL_WIDTH, PETAL_LENGTH)
final_petal_width_K2, final_petal_length_K2 = get_mean(infoK2, -1, PETAL_WIDTH, PETAL_LENGTH)

# Initialise the subplot function using number of rows and columns
figure, axis = plt.subplots(2, 3)
figure.suptitle("Starting, Intermidiate, and Converged Clusters for K = 2 and K = 3", fontsize=16)
axis[0, 0].plot(all_petal_widths, all_petal_lengths, 'o', color='black', label='Petal')
axis[0, 0].plot(starting_petal_width_K2, starting_petal_length_K2, 'r^', label="Means")
axis[0,0].set_title("K = 2 Starting")
axis[0,0].set_xlabel('Petal Width')
axis[0,0].set_ylabel('Petal Length') 
axis[0,0].legend()

axis[0, 1].plot(all_petal_widths, all_petal_lengths, 'o', color='black', label='Petal')
axis[0, 1].plot(mid_petal_width_K2, mid_petal_length_K2, 'r^', label="Means after 1 iteration")
axis[0,1].set_title("K = 2 Intermidiate")

axis[0, 2].plot(all_petal_widths, all_petal_lengths, 'o', color='black', label='Petal')
axis[0, 2].plot(final_petal_width_K2, final_petal_length_K2 , 'r^', label="Means after convergence")
axis[0,2].set_title("K = 2 Converged")

#Run kMeansClustering for K=3, with a max of 50 iterations
meansK3, infoK3 = kMeansClustering(data, 3, 50)

#Get the starting, intermediate, and converged means for k = 3
starting_petal_width_K3, starting_petal_length_K3 = get_mean(infoK3, 0, PETAL_WIDTH, PETAL_LENGTH)
mid_petal_width_K3, mid_petal_length_K3 = get_mean(infoK3, 1, PETAL_WIDTH, PETAL_LENGTH)
final_petal_width_K3, final_petal_length_K3 = get_mean(infoK3, -1, PETAL_WIDTH, PETAL_LENGTH)

# Initialise the subplot function using number of rows and columns
axis[1,0].plot(all_petal_widths , all_petal_lengths , 'o' , color='black' , label='Petal')
axis[1,0].plot(starting_petal_width_K3 , starting_petal_length_K3 , 'r^' , label="Starting Means")
axis[1,0].set_title("K = 3, Starting")

axis[1,1].plot(all_petal_widths , all_petal_lengths , 'o' , color='black' , label='Petal')
axis[1,1].plot(mid_petal_width_K3 , mid_petal_length_K3 , 'r^' , label="Means after 1 iteration")
axis[1,1].set_title("K = 3, Intermidiate")
axis[1,1].set_xlabel('Petal Width')
axis[1,1].set_ylabel('Petal Length')

axis[1,2].plot(all_petal_widths , all_petal_lengths , 'o' , color='black' , label='Petal')
axis[1,2].plot(final_petal_width_K3 , final_petal_length_K3 , 'r^' , label="Means after convergence")
axis[1,2].set_title("K = 3, Converged")

#Set a large size
figure.set_size_inches(12, 9)

plt.show()
