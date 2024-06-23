import pandas as pd
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
combinations = list(itertools.product(numerical_combinations.keys(), categorical_combinations.keys()))

# Создание и сохранение новых комбинированных датасетов
os.makedirs('processed_data/combinations', exist_ok=True)

for num_method, cat_method in combinations:
    combined_data = pd.concat([numerical_combinations[num_method], categorical_combinations[cat_method]], axis=1)
    combined_data.to_csv(f'processed_data/combinations/{num_method}_{cat_method}_data.csv', index=False)
