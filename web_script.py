import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

scaler = joblib.load('scaler.joblib')
label_encoder = joblib.load('label_encoder.joblib')
one_hot_encoder = joblib.load('one_hot_encoder.joblib')

model = joblib.load('trained_model.joblib')

airlines = ['Airline_A', 'Airline_B', 'Airline_C']
airports = ['JFK', 'LAX', 'ORD', 'DFW', 'DEN']
days_of_week = ['Понедельник', 'Вторник', 'Среда', 'Четверг', 'Пятница', 'Суббота', 'Воскресенье']
months = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь', 'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь']
time_of_day = list(range(24))

# Создание заголовка
st.title('Определение задержки рейса')

# Выпадающий список для выбора авиакомпании
selected_airline = st.selectbox('Авиакомпания', airlines)

# Выпадающий список для выбора аэропорта вылета
selected_departure_airport = st.selectbox('Аэропорт вылета', airports)

# Выпадающий список для выбора аэропорта прилета
selected_arrival_airport = st.selectbox('Аэропорт прилета', airports)

# Выпадающий список для выбора дня недели
selected_day = st.selectbox('День недели', days_of_week)

# Выпадающий список для выбора месяца года
selected_month = st.selectbox('Месяц', months)

# Выпадающий список для выбора времени вылета
selected_time_of_departure = st.selectbox('Время вылета', time_of_day)

# Выпадающий список для выбора времени прилёта
selected_time_of_arrival = st.selectbox('Время прилета', time_of_day)     

# Вводим данные о dhtvtyb gjk`nf:
time_of_flight = st.number_input('Продолжительность полета:', min_value=0)


num_data = pd.DataFrame([ (selected_time_of_departure, selected_time_of_arrival, time_of_flight) ])
cat_data = pd.DataFrame([(selected_airline, selected_departure_airport, selected_arrival_airport, selected_day,selected_month)])

num_data_scaled = scaler.transform(num_data)
cat_data_label = cat_data.apply(lambda col: label_encoder[col.name].transform(col))

input_data = pd.DataFrame([(cat_data_label, num_data_scaled)])

prediction = model.predict(input_data)

# Отображение результата:
st.write('Задержка:', 'Да' if prediction[0] else 'Нет')
