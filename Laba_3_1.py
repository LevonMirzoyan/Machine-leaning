import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
# 1
columns = [
    "Class",                   # создаем название блоков
    "Alcohol",
    "Malic acid",
    "Ash",
    "Alcalinity of ash",
    "Magnesium",
    "Total phenols",
    "Flavanoids",
    "Nonflavanoid phenols",
    "Proanthocyanins",
    "Color intensity",
    "Hue",
    "OD280/OD315 of diluted wines",
    "Proline",
]

data = pd.read_csv("wine.data", index_col=False, names=columns) #читаем файл, делаем так чтоб не возвращалось имя столбца
# 2                                                             #делаем заголовки колонок из переменной columns
X = data.loc[:, data.columns != "Class"] # записываем в переменную все кроме Class
y = data["Class"]  # записываем в переменную Class (выделяем Class)

# 3
cv = KFold(n_splits=5, shuffle=True, random_state=42) # метод кросс валидации по 5 блоков shuffle отвечает за перемещивание
# 4                                                   # KFold создаем генератор с random_state=42
best_score = 0
best_k = 0  # это кол -во итераций
for k in range(1, 51):  #  заводим цикл от 0 до 51
    knn = KNeighborsClassifier(n_neighbors=k)  # создаем свою классификацию и задаем кол-во соседей k

    score = cross_val_score(knn, X, y, cv=cv, scoring="accuracy").mean() # делаем оценку нашей крос валидации и усредняем ее

    if score > best_score:           #  потом каждое значение проверяем и выводим наибольшее
        best_score, best_k = score,k

print(best_k)
print(f"{best_score:.2f}")

# 5
best_score_ = 0
best_k_ = 0                       # проводим маштабирование признаков
for k in range(1, 51):                  # как и в пункте 4
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, scale(X), y, cv=cv, scoring="accuracy").mean()

    if score > best_score_:                          # ищем оптимальный параметр
        best_score_, best_k_ = score, k

print(best_k_)                                         # выводим его
print(f"{best_score_:.2f}")

