import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import itertools
import os

# Загрузка обработанных данных
standardized_numerical_data = pd.read_csv('processed_data/numerical/standardized/standardized_data.csv')
normalized_numerical_data = pd.read_csv('processed_data/numerical/normalized/normalized_data.csv')
label_encoded_categorical_data = pd.read_csv('processed_data/categorical/label_encoded/label_encoded_data.csv')
one_hot_encoded_categorical_data = pd.read_csv('processed_data/categorical/one_hot_encoded/one_hot_encoded_data.csv')

# Определение возможных комбинаций
numerical_combinations = {
    'standardized': standardized_numerical_data,
    'normalized': normalized_numerical_data
}

categorical_combinations = {
    'label_encoded': label_encoded_categorical_data,
    'one_hot_encoded': one_hot_encoded_categorical_data
}

# Генерация различных комбинаций
combinations = list(itertools.product(categorical_combinations.keys(), numerical_combinations.keys()))

# Функция для обучения и тестирования модели на конкретном датасете
def train_and_evaluate_model(dataset_path):
    data = pd.read_csv(dataset_path)

    data_0 = pd.read_csv('clean_flights_data.csv')

    data_0['Задержка'] = data_0['Задержка прилета'].apply(lambda x: 1 if x > 15 else 0)
    data_0 = data_0.drop(columns=['Задержка прилета'])
    y = data_0['Задержка']

    # Разделение данных на признаки и целевую переменную
    X = data

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучение модели
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Оценка модели
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

# Оценка всех созданных датасетов
best_accuracy = 0
best_dataset_path = ''

for cat_method, num_method in combinations:
    dataset_path = f'processed_data/combinations/{cat_method}_{num_method}_data.csv'
    accuracy = train_and_evaluate_model(dataset_path)

    print(f'Accuracy for {cat_method}_{num_method} data: {accuracy}')

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_dataset_path = dataset_path

print(f'Best dataset for training: {best_dataset_path} with accuracy: {best_accuracy}')
