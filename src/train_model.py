import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def create_timeseries_data(data, window_size):
    i = 1
    while i < window_size:
        data[f"co2_{i}"] = data["co2"].shift(-i)
        i += 1
    data["target"] = data["co2"].shift(-i)
    data = data.dropna(axis=0)
    return data

data = pd.read_csv("C:/Users/Hoang Le/PycharmProjects/Time Forecasting/co2.csv")
data["time"] = pd.to_datetime(data["time"], yearfirst=True)
data["co2"] = data["co2"].interpolate()

window_size = 5
train_ratio = 0.8
data = create_timeseries_data(data, window_size)
x = data.drop(["time", "target"], axis=1)
y = data["target"]

num_samples = len(x)
x_train = x[:int(num_samples * train_ratio)]
y_train = y[:int(num_samples * train_ratio)]
x_test = x[int(num_samples * train_ratio):]
y_test = y[int(num_samples * train_ratio):]

model = LinearRegression()
model.fit(x_train, y_train)

def root_mean_squared_error(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

y_predict = model.predict(x_test)
print(f"MAE: {mean_absolute_error(y_test, y_predict)}")
print(f"MSE: {mean_squared_error(y_test, y_predict)}")
print(f"RMSE: {root_mean_squared_error(y_test, y_predict)}")
print(f"R2: {r2_score(y_test, y_predict)}")

input_data = [362, 362.3, 362.6, 363, 363.1]
for _ in range(10):
    y_predict = model.predict([input_data])
    print(f"Input: {input_data}. Prediction: {y_predict}")
    input_data = input_data[1:] + y_predict.tolist()
