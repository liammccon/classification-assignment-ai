from .data import iris
#iris = datasets.load_iris()

SEPAL_LENGTH = 0
SEPAL_WIDTH = 1
PETAL_LENGTH = 2
PETAL_WIDTH = 3

SETOSA = 0
VERSICOLOR = 1
VIRGINICA = 2


import matplotlib.pyplot as plt
import numpy as np

#Getting subsets of the data
sepal_lengths = iris['data'][:,SEPAL_LENGTH]
sepal_widths = iris['data'][:,SEPAL_WIDTH]
petal_lengths = iris['data'][:,PETAL_LENGTH]
petal_widths = iris['data'][:,PETAL_WIDTH]

target = iris['target']
VIRGINICA_START = np.where(target == VIRGINICA)[0][0] #Get the index of the first virginica flower in the sample
VIRGINICA_END = VIRGINICA_START + len(np.where(target == VIRGINICA)[0])

SETOSA_START = np.where(target == SETOSA)[0][0] # Get the index of the first setosa flower in the sample
SETOSA_END = SETOSA_START + len(np.where(target == SETOSA)[0]) # Get the index of the last setosa flower in the sample

VERSICOLOR_START = np.where(target == VERSICOLOR)[0][0] # Get the index of the first versicolor flower in the sample
VERSICOLOR_END = VERSICOLOR_START + len(np.where(target == VERSICOLOR)[0]) # Get the index of the last versicolor flower in the sample

setosa_sepal_lengths = sepal_lengths[SETOSA_START:SETOSA_END]
setosa_sepal_widths = sepal_widths[SETOSA_START:SETOSA_END]
setosa_petal_lengths = petal_lengths[SETOSA_START:SETOSA_END]
setosa_petal_widths = petal_widths[SETOSA_START:SETOSA_END]

versicolor_sepal_lengths = sepal_lengths[VERSICOLOR_START:VERSICOLOR_END]
versicolor_sepal_widths = sepal_widths[VERSICOLOR_START:VERSICOLOR_END]
versicolor_petal_lengths = petal_lengths[VERSICOLOR_START:VERSICOLOR_END]
versicolor_petal_widths = petal_widths[VERSICOLOR_START:VERSICOLOR_END]

virginica_sepal_lengths = sepal_lengths[VIRGINICA_START:VIRGINICA_END]
virginica_sepal_widths = sepal_widths[VIRGINICA_START:VIRGINICA_END]
virginica_petal_lengths = petal_lengths[VIRGINICA_START:VIRGINICA_END]
virginica_petal_widths = petal_widths[VIRGINICA_START:VIRGINICA_END]

def delete_me():
    print("hi")