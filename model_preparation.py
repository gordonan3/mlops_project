import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score 
import itertools 
import os 
import joblib
 
data = pd.read_csv('processed_data/combinations/standardized_label_encoded_data.csv') 
 
# Разделение данных на признаки и целевую переменную 
X = data.drop(columns=['Задержка']) 
y = data['Задержка']

y = y.apply(lambda x: 1 if x > 0 else 0)
 
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
