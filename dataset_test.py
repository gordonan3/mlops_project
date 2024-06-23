import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import itertools
import os

# Функция для обучения и тестирования модели на конкретном датасете
def train_and_evaluate_model(dataset_path):
    data = pd.read_csv(dataset_path)
    
    # Разделение данных на признаки и целевую переменную
    X = data.drop(columns=['Задержка'])
    y = data['Задержка']
    
     # Преобразование целевой переменной в бинарные метки (0 или 1)
    y = y.apply(lambda x: 1 if x > 0 else 0)
    
    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Обучение модели
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Оценка модели
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

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
combinations = list(itertools.product(numerical_combinations.keys(), categorical_combinations.keys()))

# Создание и сохранение новых комбинированных датасетов
os.makedirs('processed_data/combinations', exist_ok=True)

for num_method, cat_method in combinations:
    combined_data = pd.concat([numerical_combinations[num_method], categorical_combinations[cat_method]], axis=1)
    combined_data.to_csv(f'processed_data/combinations/{num_method}_{cat_method}_data.csv', index=False)

print("Новые комбинированные датасеты успешно созданы и сохранены.")

# Оценка всех созданных датасетов
best_accuracy = 0
best_dataset_path = ''

for num_method, cat_method in combinations:
    dataset_path = f'processed_data/combinations/{num_method}_{cat_method}_data.csv'
    accuracy = train_and_evaluate_model(dataset_path)
    
    print(f'Accuracy for {num_method}_{cat_method} data: {accuracy}')
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_dataset_path = dataset_path

print(f'Best dataset for training: {best_dataset_path} with accuracy: {best_accuracy}')
