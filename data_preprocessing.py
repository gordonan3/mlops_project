import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
import joblib

# Создание папок для хранения обработанных данных
os.makedirs('processed_data/numerical/standardized', exist_ok=True)
os.makedirs('processed_data/numerical/normalized', exist_ok=True)
os.makedirs('processed_data/categorical/label_encoded', exist_ok=True)
os.makedirs('processed_data/categorical/one_hot_encoded', exist_ok=True)

# Загрузка данных
data = pd.read_csv('clean_flights_data.csv')

data['Задержка'] = data['Задержка прилета'].apply(lambda x: 1 if x > 15 else 0)
data = data.drop(columns=['Задержка прилета'])
y = data['Задержка']

# Определение численных и категориальных колонок
cat_columns = []
num_columns = []

for column_name in data.columns:
    if data[column_name].dtypes == object:
        cat_columns.append(column_name)
    else:
        num_columns.append(column_name)

# Удаление колонки 'Задержка' из числовых колонок
num_columns.remove('Задержка')

# Обработка числовых данных
numerical_data = data[num_columns]

# Стандартизация числовых данных
scaler = StandardScaler()
standardized_numerical_data = pd.DataFrame(scaler.fit_transform(numerical_data), columns=num_columns)
standardized_numerical_data.to_csv('processed_data/numerical/standardized/standardized_data.csv', index=False)

# Нормализация числовых данных
normalizer = MinMaxScaler()
normalized_numerical_data = pd.DataFrame(normalizer.fit_transform(numerical_data), columns=num_columns)
normalized_numerical_data.to_csv('processed_data/numerical/normalized/normalized_data.csv', index=False)

# Обработка категориальных данных
categorical_data = data[cat_columns]

# Сохранение label encoders для каждой категориальной переменной
label_encoder = {}
for column in cat_columns:
    le = LabelEncoder()
    categorical_data[column] = le.fit_transform(categorical_data[column])
    label_encoder[column] = le
categorical_data.to_csv('processed_data/categorical/label_encoded/label_encoded_data.csv', index=False)

# One-Hot Encoding
one_hot_encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded_categorical_data = pd.DataFrame(one_hot_encoder.fit_transform(categorical_data), columns=one_hot_encoder.get_feature_names_out(cat_columns))
one_hot_encoded_categorical_data.to_csv('processed_data/categorical/one_hot_encoded/one_hot_encoded_data.csv', index=False)

# Сохранение обученного scaler и label encoder
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(label_encoder, 'label_encoder.joblib')
joblib.dump(normalizer, 'normalizer.joblib')
joblib.dump(one_hot_encoder, 'one_hot_encoder.joblib')
