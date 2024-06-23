import numpy as np
import pandas as pd

# Функция для генерации данных о рейсах
def generate_flight_data(num_samples, noise=False):
    np.random.seed(42)

    # Генерация случайных данных
    airlines = ['Airline_A', 'Airline_B', 'Airline_C']
    airports = ['JFK', 'LAX', 'ORD', 'DFW', 'DEN']
    days_of_week = ['Понедельник', 'Вторник', 'Среда', 'Четверг', 'Пятница', 'Суббота', 'Воскресенье']
    months = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь', 'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь']


    airline = np.random.choice(airlines, num_samples)
    origin_airport = np.random.choice(airports, num_samples)
    destination_airport = np.random.choice(airports, num_samples)
    day_of_week = np.random.choice(days_of_week, num_samples)
    month = np.random.choice(months, num_samples)
    departure_time = np.random.randint(0, 24, num_samples)
    arrival_time = (departure_time + np.random.randint(1, 6, num_samples)) % 24
    flight_duration = np.random.randint(1, 6, num_samples) + np.random.rand(num_samples)

    # Задержка рейса (в минутах)
    arrival_delay = np.random.normal(0, 30, num_samples)  # Средняя задержка 0 минут с отклонением 30 минут

    # Добавление шума в данные
    if noise:
        arrival_delay += np.random.normal(0, 100, num_samples)  # Дополнительный шум

    # Создание DataFrame
    data = pd.DataFrame({
        'Авиакомпания': airline,
        'Аэропорт вылета': origin_airport,
        'Аэропорт прилета': destination_airport,
        'День недели': day_of_week,
        'Месяц': month,
        'Время вылета': departure_time,
        'Время прилета': arrival_time,
        'Продолжительность полета': flight_duration,
        'Задержка прилета': arrival_delay
    })

    return data

clean_data = generate_flight_data(1000, noise=False)
clean_data.to_csv('clean_flights_data.csv', index=False)

# Генерация данных с шумом
noisy_data = generate_flight_data(1000, noise=True)
noisy_data.to_csv('noisy_flights_data.csv', index=False)