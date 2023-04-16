'''Neural Network, Question E:
Write code that computes the summed gradient for an ensemble of patterns. Illustrate
the gradient by showing (i.e. plotting) how the decision boundary changes for a small
step. (10 marks)
'''

from neural_network_a import *

BIAS = -1

def summed_gradient(weights:list, bias:float, chosen_weight:int, two_dimension_subset: tuple = None):
    '''
    Calculates the summed gradient for a given weight, using classes Versicolor and Virginica. 
    Returns sum over all N samples of -2(y - a)*sig(z)*(1 - sig(z))*xi, where \n
    y is the correct label of sample n,\n 
    a is the output of the neural network,\n 
    z is w1*x1 + w2*x2 + w3*x3 + w4*x4 + w0.
    chosen_weight can be 0 to 4 to indicate the weights corresponding with dimensions x0, x1, x2, x3  OR
    chosen_weight can be -1 to indicate bias
    '''
    data = iris['data']
    class_0_vers_start, class_0_vers_end = VERSICOLOR_START, VERSICOLOR_END
    class_1_virg_start, class_1_virg_end = VIRGINICA_START, VIRGINICA_END
    
    versicolor_0_data = data[class_0_vers_start:class_0_vers_end]
    virginica_1_data = data[class_1_virg_start:class_1_virg_end]
    sum = 0

    for sample in versicolor_0_data:
        dLdAforSample = dLdA(sample, 0, weights, bias, two_dimension_subset)
        dAdWforSample = dAdW(sample, weights, bias, chosen_weight, two_dimension_subset)
        sum += dLdAforSample * dAdWforSample
        

    for sample in virginica_1_data:
        dLdAforSample = dLdA(sample, 1, weights, bias, two_dimension_subset)
        dAdWforSample = dAdW(sample, weights, bias, chosen_weight, two_dimension_subset)
        sum += dLdAforSample * dAdWforSample
    
    return sum

def dLdA(sample, y, weights, bias, two_dimension_subset):
    ''' Get the value of dL/dA for a given sample, which is -2(yn - a5n), 
    yn being the label for the sample, 
    a5n being the output of the neural network
    '''
    if two_dimension_subset == None:
        return -2 * (y - one_layer_network_4D(sample, weights, bias))
    else:
        return -2 * (y - one_layer_network_2D(sample, weights, bias, two_dimension_subset))

def dAdW(sample, weights, bias, x_feature, two_dimension_subset):
    '''Get the vlue of dL/dWi for a given sample and feature/weight i.
    Returns sig(z)*(1 - sig(z))*x[i], where z = x[0]*w[0] + ... + x[3]w[3] + bias
    '''
    if two_dimension_subset == None:
        return dAdW4D(sample, weights, bias, x_feature)
    else: 
        return dAdW2D(sample, weights, bias, x_feature, two_dimension_subset)
    
def dAdW4D(sample, weights, bias, x_feature):
    z = 0
    for feature in range(4):
        z += sample[feature]*weights[feature]
    z += bias
    sigZ = sig(z)
    product = sigZ * (1 - sigZ)
    if x_feature == BIAS:
        return product
    else: return  product * sample[x_feature]

def dAdW2D(sample, weights, bias, x_feature, two_dimension_subset):
    z = sample[two_dimension_subset[0]]*weights[0] + sample[two_dimension_subset[1]]*weights[1] + bias
    sigZ = sig(z)
    product = sigZ * (1 - sigZ)
    if x_feature == BIAS:
        return product
    else: return  product * sample[x_feature]




#Plotting the petal widths and lenghts for Versicolor and Virginica:
figure, axis = plt.subplots()
figure.suptitle("Updating Decision Boundary", fontsize=16)
axis.plot( versicolor_petal_widths, versicolor_petal_lengths, 'o', color='blue', label='Versicolor')
axis.plot( virginica_petal_widths, virginica_petal_lengths, 'o', color='green', label='Virginica')

axis.set_ylabel('Petal Length') 

#Plotting the decision boundaries
def get_decision_line(x, w1:float, w2:float, bias):
    ''' 
    Returns a y = mx + b form of the decision boundary derived from 0.5 = sig(x*w1 + y*w2 + bias)
    '''
    return (-x * w1 - bias) / w2

x = np.linspace(1,2.6,100) #Making the x values for the line
w1, w2, bias = 0, 1, -6 #Setting the first weights to the "bad" weights from question B
iterations = 4
two_dimensions = (PETAL_WIDTH, PETAL_LENGTH)
alpha = .0015 #alpha chosen based on trying different values on this plot
colors = ['#a83232', '#a85532', '#a87d32', '#a8a432', '#79a832', '#50a832', '#32a846']

#Create info, for seeing all the collected data
info = {
    'w1': [],
    'w2': [],
    'bias': [],
    'mse': []
}

def update_info(info, w1, w2, bias, mse):
    info['w1'].append(w1)
    info['w2'].append(w2)
    info['bias'].append(bias)
    info['mse'].append(mse)

#Get and plot first values of y and mean squared error
y = get_decision_line(x, w1, w2, bias)
mse = mean_sq_error([w1, w2], bias, VERSICOLOR, VIRGINICA, two_dimensions)
update_info(info, w1, w2, bias, mse)
axis.plot(x,y, color = colors[0], label=f'Decision Boundary Iteration: 0\nMean Square Error: {mse:.4f}')

#Plot the next iterations of the decision boundary
for i in range(1, len(colors)):
    w1_new = w1 - alpha * summed_gradient([w1, w2], bias, PETAL_WIDTH, two_dimensions)
    w2_new = w2 - alpha * summed_gradient([w1, w2], bias, PETAL_LENGTH, two_dimensions)
    bias_new = bias - alpha * summed_gradient([w1, w2], bias, BIAS, two_dimensions)
    y_new = get_decision_line(x, w1_new, w2_new, bias_new)

    w1, w2, bias = w1_new, w2_new, bias_new

    mse = mean_sq_error([w1, w2], bias, VERSICOLOR, VIRGINICA, two_dimensions)
    axis.plot(x, y_new, color = colors[i], label=f'Iteration: {i}, MSE: {mse:.4f}')
    update_info(info, w1, w2, bias, mse)

axis.set_title(f"Starting weights: w1 = 0, w2 = 1, w0 = -6\nEnding weights: w1 = {w1:.3f}, w2 = {w2:.3f}, w0 = {bias:.3f}")
axis.set_xlabel(f'Petal Width')
axis.legend()

#Set a large size
figure.set_size_inches(10, 7)
plt.show()