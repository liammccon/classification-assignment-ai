# Instalations
## `matplotlib`
A library used for plotting and graphing
## `Scikit-learn`
A convinient which has the full Iris dataset. Install with:
```
$ pip install -U scikit-learn
```
### Iris Data Set
Get the iris data set with `iris = datasets.load_iris()`

The data is stored in this format:
```
iris = {'data':  array([[5.1, 3.5, 1.4, 0.2],
                        [4.9, 3. , 1.4, 0.2],
                        [4.7, 3.2, 1.3, 0.2],
                        [4.6, 3.1, 1.5, 0.2],...
'target': array([0, 0, 0, ... 1, 1, 1, ... 2, 2, 2, ...
'target_names': array(['setosa', 'versicolor', 'virginica'], dtype='<U10'), 
...}
```
For `'data': array([[5.1, 3.5, 1.4, 0.2], ... ])`
- Each entry in the array stores: 
`[Sepal length, Sepal width, Petal length, Petal width]`

For `'target': array([0, 0, ... 2, 2])`
- Each entry in the array stores the target flower type for the corresponding entry in `'data'`
    - `0`: `setosa`
    - `1`: `versicolor`
    - `2`: `virginica`
"""


deleteme
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Voronoi.html - create a voronoi diagram

#BUG? When no points are assigned to a cluster, its mean becomes [0, 0, 0, ...] and no points will be assigned -> minimum
    #Solution - when the mean is 0, just generate a new random mean - still doesnt help when one mean takes everything
        #Solution - only allow a certain % of the points to be assigned to one cluster
        #OR Better solution: start with a large K number, then remove all means which have no membership
    #TRUE SOLUTION: Select the random means from the data! Guarantees >= 1 point per mean

#Returns a tuple (means, info) where means is the final means and info contains information about the means and distortion for each iteration
