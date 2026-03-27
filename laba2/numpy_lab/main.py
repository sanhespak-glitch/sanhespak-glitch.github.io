# 
# Лабораторная работа: Численные вычисления и анализ данных с использованием NumPy

# Формат выполнения: самостоятельная работа.

# Перед началом:
# 1. Создайте виртуальное окружение:
#    python -m venv numpy_env
   
# 2. Активируйте виртуальное окружение:
#    - Windows: numpy_env\Scripts\activate
#    - Linux/Mac: source numpy_env/bin/activate
   
# 3. Установите зависимости:
#    pip install numpy matplotlib seaborn pandas pytest

# Структура проекта:

# numpy_lab/
# ├── main.py
# ├── test.py
# ├── data/
# │   └── students_scores.csv
# └── plots/

# В папке data создайте файл students_scores.csv со следующим содержимым:

# math,physics,informatics
# 78,81,90
# 85,89,88
# 92,94,95
# 70,75,72
# 88,84,91
# 95,99,98
# 60,65,70
# 73,70,68
# 84,86,85
# 90,93,92

# (Дополнительно можно использовать публичные датасеты Kaggle:
# Students Performance Dataset:
# https://www.kaggle.com/datasets/spscientist/students-performance-in-exams
# или любой аналогичный табличный CSV)

# Задача: реализовать все функции, чтобы проходили тесты.
# 

import os
import numpy as np
from typing import Union
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. СОЗДАНИЕ И ОБРАБОТКА МАССИВОВ
# ============================================================

def create_vector() -> np.ndarray:
    """
    Создать массив от 0 до 9.

    Изучить:
    https://numpy.org/doc/stable/reference/generated/numpy.arange.html
    
    Returns:
        numpy.ndarray: Массив чисел от 0 до 9 включительно
    """
    return np.arange(10)


def create_matrix()-> np.ndarray:
    """
    Создать матрицу 5x5 со случайными числами [0,1].

    Изучить:
    https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html
    
    Returns:
        numpy.ndarray: Матрица 5x5 со случайными значениями от 0 до 1
    """
    return np.random.rand(5,5)


def reshape_vector(vec: np.ndarray) -> np.ndarray:
    """
    Преобразовать (10,) -> (2,5)

    Изучить:
    https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
    
    Args:
        vec (numpy.ndarray): Входной массив формы (10,)
    
    Returns:
        numpy.ndarray: Преобразованный массив формы (2, 5)
    """
    return vec.reshape(2,5)


def transpose_matrix(mat: np.ndarray) -> np.ndarray:
    """
    Транспонирование матрицы.

    Изучить:
    https://numpy.org/doc/stable/reference/generated/numpy.transpose.html
    
    Args:
        mat (numpy.ndarray): Входная матрица
    
    Returns:
        numpy.ndarray: Транспонированная матрица
    """
    return np.transpose(mat)


# ============================================================
# 2. ВЕКТОРНЫЕ ОПЕРАЦИИ
# ============================================================

def vector_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Сложение векторов одинаковой длины.
    (Векторизация без циклов)
    
    Args:
        a (numpy.ndarray): Первый вектор
        b (numpy.ndarray): Второй вектор
    
    Returns:
        numpy.ndarray: Результат поэлементного сложения
    """
    return a+b


def scalar_multiply(vec: np.ndarray, scalar: Union[float, int]) -> np.ndarray:
    """
    Умножение вектора на число.
    
    Args:
        vec (numpy.ndarray): Входной вектор
        scalar (float/int): Число для умножения
    
    Returns:
        numpy.ndarray: Результат умножения вектора на скаляр
    """
    return vec*scalar
    pass


def elementwise_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Поэлементное умножение.
    
    Args:
        a (numpy.ndarray): Первый вектор/матрица
        b (numpy.ndarray): Второй вектор/матрица
    
    Returns:
        numpy.ndarray: Результат поэлементного умножения
    """
    return a*b
    pass


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """
    Скалярное произведение.

    Изучить:
    https://numpy.org/doc/stable/reference/generated/numpy.dot.html
    
    Args:
        a (numpy.ndarray): Первый вектор
        b (numpy.ndarray): Второй вектор
    
    Returns:
        float: Скалярное произведение векторов
    """
    return np.dot(a, b)
    


# ============================================================
# 3. МАТРИЧНЫЕ ОПЕРАЦИИ
# ============================================================

def matrix_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Умножение матриц.

    Изучить:
    https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
    
    Args:
        a (numpy.ndarray): Первая матрица
        b (numpy.ndarray): Вторая матрица
    
    Returns:
        numpy.ndarray: Результат умножения матриц
    """
    return np.matmul(a, b)


def matrix_determinant(a: np.ndarray) -> float:
    """
    Определитель матрицы.

    Изучить:
    https://numpy.org/doc/stable/reference/generated/numpy.linalg.det.html
    
    Args:
        a (numpy.ndarray): Квадратная матрица
    
    Returns:
        float: Определитель матрицы
    """
    return np.linalg.det(a)


def matrix_inverse(a: np.ndarray) -> np.ndarray:
    """
    Обратная матрица.

    Изучить:
    https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html
    
    Args:
        a (numpy.ndarray): Квадратная матрица
    
    Returns:
        numpy.ndarray: Обратная матрица
    """
    return np.linalg.inv(a)


def solve_linear_system(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Решить систему Ax = b

    Изучить:
    https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html
    
    Args:
        a (numpy.ndarray): Матрица коэффициентов A
        b (numpy.ndarray): Вектор свободных членов b
    
    Returns:
        numpy.ndarray: Решение системы x
    """
    return np.linalg.solve(a, b)


# ============================================================
# 4. СТАТИСТИЧЕСКИЙ АНАЛИЗ
# ============================================================

def load_dataset(path="data/students_scores.csv") -> np.ndarray:
    """
    Загрузить CSV и вернуть NumPy массив.
    
    Args:
        path (str): Путь к CSV файлу
    
    Returns:
        numpy.ndarray: Загруженные данные в виде массива
    """
    return pd.read_csv(path).to_numpy()


def statistical_analysis(data: np.ndarray) -> dict:
    """
    Представьте, что данные — это результаты экзамена по математике.
    Нужно оценить:
    - средний балл
    - медиану
    - стандартное отклонение
    - минимум
    - максимум
    - 25 и 75 перцентили

    Изучить:
    https://numpy.org/doc/stable/reference/generated/numpy.mean.html
    https://numpy.org/doc/stable/reference/generated/numpy.median.html
    https://numpy.org/doc/stable/reference/generated/numpy.std.html
    https://numpy.org/doc/stable/reference/generated/numpy.percentile.html
    
    Args:
        data (numpy.ndarray): Одномерный массив данных
    
    Returns:
        dict: Словарь со статистическими показателями
    """
    return {"mean": np.mean(data),
        "median": np.median(data),
        "std": np.std(data),
        "min": np.min(data),
        "max": np.max(data),
        "percentile_25": np.percentile(data, 25),
        "percentile_75": np.percentile(data, 75)}

def normalize_data(data: np.ndarray) -> np.ndarray:
    """
    Min-Max нормализация.
    
    Формула: (x - min) / (max - min)
    
    Args:
        data (numpy.ndarray): Входной массив данных
    
    Returns:
        numpy.ndarray: Нормализованный массив данных в диапазоне [0, 1]
    """
    return (data -np.min(data)) / (np.max(data) - np.min(data))


# ============================================================
# 5. ВИЗУАЛИЗАЦИЯ
# ============================================================

def plot_histogram(data: np.ndarray):
    """
    Построить гистограмму распределения оценок по математике.

    Изучить:
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html
    
    Args:
        data (numpy.ndarray): Данные для гистограммы
    """
    plt.hist(data)
    plt.title("гистограмма распределения оценок по математике")
    plt.xlabel("оценка")
    plt.ylabel("студнеты")
    plt.savefig("./plots/gistogramma.png")
    plt.close()

# plot_histogram(load_dataset("./data/students_scores.csv")[:, 0])


def plot_heatmap(matrix: np.ndarray):
    """
    Построить тепловую карту корреляции предметов.

    Изучить:
    https://seaborn.pydata.org/generated/seaborn.heatmap.html
    
    Args:
        matrix (numpy.ndarray): Матрица корреляции
    """
    sns.heatmap(matrix, annot=True, cmap="viridis")
    plt.title("епловая карта корреляции предметов")
    plt.savefig("./plots/heatmap.png")
    plt.close()

corr = np.corrcoef(load_dataset("./data/students_scores.csv"), rowvar=False)
# plot_heatmap(corr)

def plot_line(x: np.ndarray, y: np.ndarray):
    """
    Построить график зависимости: студент -> оценка по математике.

    Изучить:
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
    
    Args:
        x (numpy.ndarray): Номера студентов
        y (numpy.ndarray): Оценки студентов
    """
    plt.plot(x, y, marker="o")
    plt.title("график зависимости: студент -> оценка по математике")
    plt.xlabel("Номера студентов")
    plt.ylabel("Оценки студентов")
    plt.savefig("./plots/line.png")
    plt.close()
x = np.arange(1, len(load_dataset("./data/students_scores.csv")) + 1)
y = load_dataset("./data/students_scores.csv")[:, 0]
# plot_line(x, y)
# ============================================================
# ========================== ТЕСТЫ ===========================
# ============================================================

if __name__ == "__main__":
    print("Запустите python3 -m pytest test.py -v для проверки лабораторной работы.")
