'''Compute the mean squared error for two different settings of the weights (i.e. two
different decision boundaries). Select these by hand and choose settings that give large
and small errors respectively. Plot both boundaries on the dataset. You will only use
the 2nd and 3rd iris classes in this problem. (5 marks)
'''

from iris.subsets import *
from neural_network_a import *
from k_means import *
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

#Plotting the petal widths and lenghts for Versicolor and Virginica:
figure, axis = plt.subplots(1, 2)
figure.suptitle("Title", fontsize=16)
axis[0].plot( versicolor_petal_widths, versicolor_petal_lengths, 'o', color='blue', label='Versicolor')
axis[0].plot( virginica_petal_widths, virginica_petal_lengths, 'o', color='green', label='Virginica')

axis[0].set_title("High Loss Weights")
axis[0].set_xlabel('Petal Width')
axis[0].set_ylabel('Petal Length') 
axis[0].legend()

#Plotting bad weights
#Set dividing line to y = 0x + 6
#Equivalent line is with w1 = 0, w2 = 1, w0 = -6 in 0.5 = sig(0x + y - 6) where x is width and y is length.
x = np.linspace(1,2.5,100)
y = 0*x + 6
axis[0].plot(x,y, 'r')
mse_bad = mean_sq_error([0, 1], -6,  VERSICOLOR, VIRGINICA, (PETAL_WIDTH, PETAL_LENGTH))
print(f'MSE on bad line{mse_bad}')


axis[1].plot( versicolor_petal_widths, versicolor_petal_lengths, 'o', color='blue', label='Versicolor')
axis[1].plot( virginica_petal_widths, virginica_petal_lengths, 'o', color='green', label='Virginica')

axis[1].set_title("Low Loss Weights\nw1 = 1, w2 = 2, w0 = -6.5")
axis[1].set_xlabel('Petal Width')
axis[1].set_ylabel('Petal Length') 
axis[1].legend()

#Plotting good weights
# Two points that form a good boundary line: [width, length] (1.5, 5), (2, 4.5)
# Approximate dividing line is: y = -1x + 6.5
#Decision boundary is .5 = sig(x1*w1 + x2*w2 + w0). Replacing x1 with x and x2 with y, 
#We can find an equivalent line with weights w1 = 1, w2 = 1, w0 = -6.5 in equation 0.5 = sig(x + y - 6.5)
#[3 -> 7]
x = np.linspace(1,2.5,100)
y = -x + 6.5
axis[1].plot(x,y, 'r')
print(f'MSE on good line{mean_sq_error([1, 1], -6.5,  VERSICOLOR, VIRGINICA, (PETAL_WIDTH, PETAL_LENGTH))}')
print(f'MSE on same line, flipped{mean_sq_error([1, 1], -6.5, VIRGINICA, VERSICOLOR, (PETAL_WIDTH, PETAL_LENGTH))}')

#Set a larger size
figure.set_size_inches(10, 7)
plt.show()