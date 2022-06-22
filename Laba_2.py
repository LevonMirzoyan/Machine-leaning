import pandas as pd
from sklearn.tree import DecisionTreeClassifier


# 1
data = pd.read_csv("titanic.csv", index_col="PassengerId") # читаем файл столбец PassengerId для нумерации строк
# 2
# Доступ по строковой метке
x = data.loc[:, ["Pclass", "Fare", "Age", "Sex"]] # присваиваем переменной 4 колонки

# 3
# Преобразуем строковый в числа
x["Sex"] = x["Sex"].map({"male": 0, "female": 1}) # преобразуем пол в числа 0 и 1

# 4
y = data["Survived"] # присваивание информации выжил не выжил переменной y

# 5
# Удалим пропуски
x = x.dropna() # dropna() - удаляет строки с пустыми полями
y = y[x.index]

# 6
clf = DecisionTreeClassifier(random_state=241) # Обучите решающее дерево с параметром random_state=241
clf.fit(x, y) # обучение (тренировка) модели на обучающей выборке x и y

# 7
feature_importances = pd.Series(clf.feature_importances_, index=x.columns).sort_values(ascending=False)
# feature_importances_ - возвращает вектор "важностей" признаков, sort_values(ascending=False) сортируем по убыванию, series - делает датафрайм
# columns для возврата меток столбцов
print(" ".join(feature_importances.head(2).index)) # выводим два самых важных признака( head(2) выводим первых два из списка)