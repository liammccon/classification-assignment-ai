import math
from iris.subsets import *
from clustering import distortion as d

print(math.inf > 100000000)

print(iris['data'])

print(d.distortion(iris['data']))