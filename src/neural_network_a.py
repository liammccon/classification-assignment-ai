'''Neural Network, Question A:
Write a program that calculates the mean-squared error of the iris data for simple
one-layer neural network using a sigmoid non-linearity (linear classification with sig-
moid non-linearity). The function should take three arguments: the data vectors, the
parameters defining the neural network, and the pattern classes. (10 marks)
'''
import numpy as np
from iris.data import iris
from iris.subsets import *

def mean_sq_error(query: list,
                  hidden_layer_size: int, 
                  weights_input_to_hidden:list, 
                  weights_hidden_to_output: list) -> float: 
    '''
    Calculates the mean squared error over the iris dataset for Setosa and Versicolor (the classes which are most difficult to distinguish)\n
    Given weight parameters and pattern class(es) for the one-layer neural network. \n
    Given a hidden layer size of N, the weights for input to hidden should be arranged in a N by 4 two dimensional array\n
        and the weights for hidden to output should be in an array length N\n

    Mean squared error is calculated as the sum of (y - z)^2 over the entire iris dataset (for Setosa and Versicolor), /n
    where y is the correct label (0 for Setosa and 1 for Versicolor) and z is the output of the neural network, from 0 to 1. 
    '''
    data = iris['data']
    setosa_data = data[SETOSA_START:SETOSA_END]
    versicolor_data = data[VERSICOLOR_START:VERSICOLOR_END] 
    class_1_name = "setosa"
    class_2_name = "versicolor"
    sum = 0
    for sample in setosa_data:
        prediction = neural_network(sample, hidden_layer_size, weights_input_to_hidden, weights_hidden_to_output, class_1_name, class_2_name)
        sum += (0 - prediction)**2
    
    for sample in versicolor_data:
        prediction = neural_network(sample, hidden_layer_size, weights_input_to_hidden, weights_hidden_to_output, class_1_name, class_2_name)
        sum += (1 - prediction)**2

    MSE = sum / (len(setosa_data) + len(versicolor_data))
    return MSE

def neural_network(query: list,
                  hidden_layer_size: int, 
                  weights_input_to_hidden:list, 
                  weights_hidden_to_output: list, 
                  class_1_name: str, 
                  class_2_name: str = None) -> float:
    '''
    Calculates the output of a neural network with given parameters:,\n
    Weight parameters and pattern class(es) for the one-layer neural network. \n
    Given one class, it will output the certainty (0 to 1) that the query belongs to that class\n
    Given two classes, it will output which class the query seems to belong to, class 1 (0) to class 2 (1) \n
    Given a hidden layer size of N, the weights for input to hidden should be arranged in a N by 4 two dimensional array\n
        and the weights for hidden to output should be in an array length N\n
    '''
    if len(weights_input_to_hidden) != hidden_layer_size:
        raise Exception("Given a hidden layer size of N, the weights for input to hidden should be arranged in a N by 4 two dimensional array")

    hidden_outputs = []

    #Calculating input to hidden layer 
    for neuron in range(len(hidden_layer_size)):
        dot_product = 0
        for feature in range(4):
            dot_product += query[feature] * weights_input_to_hidden[neuron][feature]
        hidden_outputs.append(sig(dot_product))

    final_dot_product = 0
    #Calculating hidden layer to output layer
    for hidden_neuron in range(len(hidden_outputs)):
        final_dot_product += hidden_outputs[hidden_neuron] * weights_hidden_to_output[hidden_neuron]
    
    output = sig(final_dot_product)

    ''' Optional print outputs
    if class_2_name == None:
        print(f"The neural network is currently {output * 100}% certain that the given input was in the class {class_1_name}")
    else:
        print(f"The neural network is {(output) * 100}% certain the query was in {class_1_name} and {(1 - output)*100}% certain the query was in class 2")
    '''
    return output

    

def sig(z):
    return 1 / (1 + np.exp(-z))

def verify_weight_arrays(hidden_layer_size: int, 
                  weights_input_to_hidden:list, 
                  weights_hidden_to_output: list) -> bool :
    if len(weights_input_to_hidden) != hidden_layer_size:
        raise Exception("Given a hidden layer size of N, the weights for input to hidden should be arranged in a N by 4 two dimensional array")
    if len(weights_input_to_hidden[0] != 4):
        raise Exception("Given a hidden layer size of N, the weights for input to hidden should be arranged in a N by 4 two dimensional array")
    if len(weights_hidden_to_output) != hidden_layer_size:
        raise Exception("The weights for hidden to output should be in an array the length of the number of neurons in the hidden layer")


