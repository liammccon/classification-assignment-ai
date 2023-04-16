'''Neural Network, Question E:
Write code that computes the summed gradient for an ensemble of patterns. Illustrate
the gradient by showing (i.e. plotting) how the decision boundary changes for a small
step. (10 marks)
'''

from neural_network_a import *

def summed_gradient(weights:list, bias:float, chosen_weight:int):
    '''
    Calculates the summed gradient for a given weight, using classes Versicolor and Virginica. 
    Returns sum over all N samples of -2(y - a)*sig(z)*(1 - sig(z))*xi, where \n
    y is the correct label of sample n,\n 
    a is the output of the neural network,\n 
    z is w1*x1 + w2*x2 + w3*x3 + w4*x4 + w0
    '''
    data = iris['data']
    class_0_vers_start, class_0_vers_end = VERSICOLOR_START, VERSICOLOR_END
    class_1_virg_start, class_1_virg_end = VIRGINICA_START, VIRGINICA_END
    
    versicolor_0_data = data[class_0_vers_start:class_0_vers_end]
    virginica_1_data = data[class_1_virg_start:class_1_virg_end]
    sum = 0

    for sample in versicolor_0_data:
        dLdAforSample = dLdA(sample, 0, weights, bias)
        dLdWforSample = dLdW(sample, weights, bias, chosen_weight)
        sum += dLdAforSample * dLdWforSample
        

    for sample in virginica_1_data:
        dLdAforSample = dLdA(sample, 1, weights, bias)
        dLdWforSample = dLdW(sample, weights, bias, chosen_weight)
        sum += dLdAforSample * dLdWforSample
    
    return sum

def dLdA(sample, y, weights, bias):
    ''' Get the value of dL/dA for a given sample, which is -2(yn - a5n), 
    yn being the label for the sample, 
    a5n being the output of the neural network
    '''
    return -2 * (y - one_layer_network_4D(sample, weights, bias))

def dLdW(sample, weights, bias, x_feature):
    z = 0
    for feature in range(4):
        z += sample[feature]*weights[feature]
    z += bias
    sigZ = sig(z)
    return sigZ * (1 - sigZ) * sample[x_feature]