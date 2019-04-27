#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

data = load_iris()

X = data["data"]
Y = data["target"]


# Antes de implementar el algoritmo centramos las muestras
# normalizando cada característica a media cero
X = X-np.mean(X, axis=0)

# PCA
# 1. Creamos un par de variables para almacenar el n de comp principales a usar
#    y el n de muestras almacenadas en nuestra matriz de diseño
k = 2
n = float(X.shape[0])

# 2. calculamos el vector de medias y la matriz de covarianza
mu = np.mean(X, axis=0)
cov = np.dot((X-mu).T, (X-mu))/(n-1)
#cov = np.cov(X.T) #Alternativamente

# 3. calcular la eigendescomposicion de la matriz de covarianza
#    y ordenar los eigenvectores en orden de eigenvalor descendiente
evals, evects = np.linalg.eig(cov)

# Lista con los indices que ordenan los eigenvalores de menor a mayor.
indices = np.argsort(evals)[::-1]

# Reordenamos
evals = evals[indices]
evects = evects.T[indices]

# componemos la matriz de proyección W concatenando como columnas los
# k primeros eigenvectores
W = np.concatenate([evects[i].reshape(-1,1) for i in range(k)], axis=1)
print(W.shape)

##############################

Xpca = np.dot(X, W)
print(Xpca.shape)

plt.scatter(Xpca[:,0], Xpca[:,1], c=Y, alpha=0.5)
plt.show()
