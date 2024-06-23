import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import itertools
import os
import joblib

data = pd.read_csv('processed_data/combinations/label_encoded_standardized_data.csv')

# Разделение данных на признаки и целевую переменную
X = data
data_0 = pd.read_csv('clean_flights_data.csv')

data_0['Задержка'] = data_0['Задержка прилета'].apply(lambda x: 1 if x > 15 else 0)

data_0 = data_0.drop(columns=['Задержка прилета'])

y = data_0['Задержка']
# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Обучение модели
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Сохраняем обученную модель
joblib.dump(model, 'trained_model.joblib')

# Оценка модели
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print('Accuracy: ', accuracy)
