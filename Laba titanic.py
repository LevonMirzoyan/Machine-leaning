import pandas as pd
import re   # модуль регулярных выражений

import sys
sys.path.append("..")


df = pd.read_csv("titanic.csv", index_col="PassengerId")
df.head()

# Какое количество мужчин и женщин ехало на корабле

sex_counts = df["Sex"].value_counts()    #считаем кол-во мужчин и женщин
print(1, f"{sex_counts['male']} {sex_counts['female']}") # выводим это количество

#Какой части пассажиров удалось выжить

survived_counts = df["Survived"].value_counts() # считаем колво виживших не выживших
survived_percent = 100.0 * survived_counts[1] / survived_counts.sum() # колво выживжих умн на 100% и делим на всех пассажиров
print(2, f"{survived_percent:.2f}") # выводим с точностью два числа после запятой

#Какую долю пассажиры первого класса составляли среди всех пассажиров

pclass_counts = df["Pclass"].value_counts() # считаем сколько пассажиров в каждом классе
pclass_percent = 100.0 * pclass_counts[1] / pclass_counts.sum() # кол-во пассажиров 1 класса умн на 100% и делим навсех пассажиров
print(3, f"{pclass_percent:.2f}") # выводим с точностью два числа после запятой

#Какого возраста были пассажиры

ages = df["Age"].dropna() # dropna() - удаляет строки с пустыми полями ( возраст = пустому месту )
print(4, f"{ages.mean():.2f} {ages.median():.2f}") # mean() - возвращает среднее значение для ages и median() - медиана для ages

#Коррелируют ли число братьев/сестер с числом родителей/детей

corr = df["SibSp"].corr(df["Parch"])  # corr() используется для нахождения попарной корреляции всех столбцов в кадре данных
print(5, f"{corr:.2f}")

#Какое самое популярное женское имя на корабле

def clean_name(name: str) -> str:
    # Первое слово до запятой - фамилия
    match = re.search("^[^,]+, (.*)", name)        # имя находиться после запятой и точки

    if match:
        name = match[1]

    # Если есть скобки - то имя пассажира в них
    match = re.search("\(([^)]+)\)", name)
    if match:
        name = match[1]

    # Удаляем обращения
    name = re.sub("(Miss\. |Mrs\. |Ms\. )", "", name) # меняем обращения на пустое поле

    # Берем первое оставшееся слово и удаляем кавычки
    name = name.split(" ")[0].replace("\"", "") # берем слово и заменяем " " на пустое поле

    return name  # возвращаем имя без фамилии и обращения

names = df[df["Sex"] == "female"]["Name"].map(clean_name)
names.head()

print(6, names.mode()[0]) # с помощью mode выбираем самое частое имя