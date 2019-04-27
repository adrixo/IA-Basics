#!/usr/bin/python

from sklearn.datasets import load_iris

data = load_iris()
Y = data["target"]
X = data["data"]


# dividimos de forma aleatoria en 2, uno para entrenar y otro evaluar
# por defecto 75%
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

# instanciamos el modelo y entrenamos
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
knn.fit(X_train, Y_train)

############################

# Predicción para la muestra X = ([ 5.5, 2.4, 3.8, 1.1])
Xnew = [[ 5.5, 2.4, 3.8, 1.1]]
print(knn.predict(Xnew))

# Tasa de acierto de nuestro algoritmo sobre el conjunto del test
print(knn.score(X_test, Y_test))
