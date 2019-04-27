#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


# Generamos un dataset con 100 muestras y una única característica
n_samples = 100
X, Y = datasets.make_regression(n_samples=n_samples, n_features=1, n_informative=1, noise=10)

# Añadimos a la matriz de diseño X la columna con las características auxiliares
X = np.concatenate([X, np.ones((n_samples,1))], axis=1)

# w = (X.T X)⁻1 X.T Y
temp = np.linalg.inv(np.dot(X.T, X))
w = np.dot(np.dot(temp, X.T), Y)
# ó
# Versión mas eficiente (se evita el uso de inv):
# X.T Xw = X.T Y
w,_,_,_ = np.linalg.lstsq(np.dot(X.T,X), np.dot(X.T, Y))

# Presentamos los puntos de nuestro dataset frente a los valores objetivo almacenados en Y
plt.scatter(X[:,0], Y)
# X[:,0] primera característica de todos los puntos en la matriz de diseño

# Salida del modelo
ys = np.array([w[0]*x + w[1] for x in range(-3, 4, 1)])
plt.plot(range(-3,4,1), ys, c="r")
plt.show()
