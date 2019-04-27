#!/usr/bin/python

from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import numpy as np

data = load_iris()
X = data["data"]
Y = data["target"]
X = X - np.mean(X, axis=0)

# Instanciamos e iniciamos
pca = PCA(n_components=2)
Xpca = pca.fit(X)

# Lo usamos
Xpca = pca.transform(X)

# Esta implementación usa el algoritmo SVD en lugar de eigendecomposition para
# Calcular los componentes principales a partir de la matriz de covarianza.
# SVD es más eficiente y estable numéricamente
# Más info: https://math.stackexchange.com/questions/3869/what-is-the-intuitive-relationship-between-svd-and-pca
