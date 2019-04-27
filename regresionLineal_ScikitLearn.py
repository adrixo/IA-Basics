#!/usr/bin/python

from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn import datasets

# Generamos los datos artificiales
#   500 muestras
#   3 características
n_samples = 500
X, Y = datasets.make_regression(n_samples=n_samples, n_features=3, n_informative=2, noise=10)

# dividimos los datos en sets de entrenamiento y test de forma aleatoria
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7)

# ENtrenamos nuestro modelo
lr = LinearRegression().fit(X_train, Y_train)

#####################################

# Predicción sobre nueva muestra no vista con anterioridad
print(lr.predict([[0.3, 1, 2]]))

# Error cuadrático medio del modelo sobre set de datos de entrenamiento y evaluacion
print( np.mean((lr.predict(X_train) - Y_train) ** 2) )
print( np.mean((lr.predict(X_test) - Y_test) ** 2) )
