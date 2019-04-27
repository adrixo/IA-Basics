#!/usr/bin/python

# Cargamos el dataset en memoria y split
from sklearn.datasets import load_digits
data = load_digits()
Y = data["target"]
X = data["data"]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

# Instanciamos nuestro modelo haciendo uso de la clase
# Vamos a evaluar una maquina de soporte vectorial lineal y otra con un kernel polinomico
from sklearn.svm import SVC, LinearSVC

svmPoly = SVC(kernel="poly", degree=2, coef0=0)
svmPoly.fit(X_train, Y_train)

lsvm = LinearSVC()
lsvm.fit(X_train, Y_train)

################################

# hacemos predicciones:
print(svmPoly.predict([X_test[0]]))
print(lsvm.predict([X_test[0]]))

print(svmPoly.score(X_test, Y_test))
print(lsvm.score(X_test, Y_test))
