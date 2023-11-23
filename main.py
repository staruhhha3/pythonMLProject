import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import numpy as np

# Загрузка данных
data = pd.read_csv('CME_JYM2022.csv')  # Замените 'your_data.csv' на реальный путь к вашим данным

# Преобразование даты
data['Date'] = pd.to_datetime(data['Date'])

# Подготовка временного ряда
time_series = data.set_index('Date')['Close'].values.reshape(-1, 1)

# Нормализация данных
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(time_series)

# Создание обучающих выборок
train_size = int(len(scaled_data) * 0.8)
train, test = scaled_data[:train_size], scaled_data[train_size:]

# Функция для создания обучающих выборок
def create_dataset(dataset, look_back=1):
    X, y = [], []
    for i in range(len(dataset)-look_back):
        X.append(dataset[i:(i+look_back), 0])
        y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(y)

# Преобразование данных для обучения
look_back = 10
X_train, y_train = create_dataset(train, look_back)
X_test, y_test = create_dataset(test, look_back)

# Инициализация модели
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Обучение модели
model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)

# Оценка производительности на тестовых данных
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Инвертирование нормализации
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])

# Графики кривых обучения, исходных данных и прогноза
plt.figure(figsize=(16, 8))

# График обучающих данных
plt.subplot(2, 1, 1)
plt.plot(y_train.flatten(), label='Исходные данные (обучение)')
plt.plot(train_predict.flatten(), label='Прогноз (обучение)')
plt.title('Прогноз на обучающих данных')
plt.legend()

# График тестовых данных
plt.subplot(2, 1, 2)
plt.plot(y_test.flatten(), label='Исходные данные (тестирование)')
plt.plot(test_predict.flatten(), label='Прогноз (тестирование)')
plt.title('Прогноз на тестовых данных')
plt.legend()

plt.tight_layout()
plt.show()


a = 3+3
print(a)
print("Тестовая работа с Git")