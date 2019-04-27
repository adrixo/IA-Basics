#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

#X = np.array( [[ 3. , 2. , 2. ], [ 2., -1., 1.]])
#print (X.shape)

######################
# Ejemplo uso graficos
X = np.random.normal(size=(2,100))

plt.scatter(X[0], X[1])
plt.show()

######################
# Ejemplo uso datasets

# 1. Desde texto
data = np.loadtxt( "datasets/iris.data", delimiter=",", dtype=np.str)

X = data[:, 0:4].astype(np.float32)
Y = data[:, -1]

Y[ Y == "Iris-setosa" ] = 0
Y[ Y == "Iris-versicolor" ] = 1
Y[ Y == "Iris-virginica" ] = 2
Y = Y.astype( np.int )

# 2. Propios de scikit-learn
from sklearn.datasets import load_iris

data = load_iris()
X = data["data"]
Y = data["target"]
print(data[ "target_names" ])

# 3. Formato LIBSVM, formato svmlight
from sklearn.datasets import load_svmlight_file

X , Y = load_svmlight_file("datasets/usps")
print(X.shape)
print(Y.shape)
X = np.asarray(X)
