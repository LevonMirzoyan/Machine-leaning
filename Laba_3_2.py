import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import scale

# 1                                        # функция которая зависит от двух параметров x и y
data = load_boston()  # загружаем выборку Boston
X = data.data     # x - признаки
y = data.target    # у - делаем целевым вектором

# 2
X = scale(X)        # маштабируем значение x

# 3
cv = KFold(n_splits=5, shuffle=True, random_state=42)  # делаем кросс валидацию

best_score = None    #  выдаем значение None т.к у нас не числовые данные
best_p = None
for p in np.linspace(1, 10, 200):    #идем по нашему одномерному массиву с измельчением от 1 до 10
    knn = KNeighborsRegressor(p=p, n_neighbors=5, weights="distance") # здесь мы производим регресс (данный параметр добавляет
                                                                     # в алгоритм веса, зависящие от расстояния до ближайших соседей)

    score = cross_val_score(knn, X, y, cv=cv, scoring="neg_mean_squared_error").mean()  #делаем оценку для каждого и усредняем

    if best_score is None or score > best_score:   # считаем и находим наибольшее
        best_score, best_p = score, p
# 4
print(f"{best_p:.2f}")

