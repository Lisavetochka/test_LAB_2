import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt

# Дані
data = {
    "Філія": np.arange(1, 13),
    "Y": [9.68, 12.02, 13.6, 13.76, 13.77, 15.1, 11.08, 12.52, 14.43, 9.91, 8.27, 9.9],
    "X1": [0.61, 1.28, 1.51, 1.59, 1.42, 1.79, 1.08, 1.24, 1.59, 0.78, 0.54, 0.85],
    "X2": [11.74, 9.01, 12.31, 11.39, 15.22, 15.42, 10.04, 13.86, 13.77, 12.51, 9.75, 10.81],
    "X3": [6.995, 7.125, 8.275, 7.805, 11.475, 10.275, 7.025, 9.255, 9.375, 7.775, 6.735, 6.825],
}

# Створення DataFrame
df = pd.DataFrame(data)

# Модель множинної регресії
X = df[["X1", "X2", "X3"]]
X = sm.add_constant(X)  # Додаємо константу
Y = df["Y"]

# Перевірка мультиколінеарності за допомогою VIF
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Вивести VIF
print("VIF для кожної змінної:")
print(vif_data)

# Якщо VIF значення більше 10, варто розглянути вилучення змінної з високою мультиколінеарністю.
# Після аналізу VIF можна вилучити змінну з високим VIF (якщо таке є).

# Створення моделі множинної регресії
model = sm.OLS(Y, X).fit()

# Підсумки моделі
print("\nПідсумки моделі множинної регресії:")
print(model.summary())

# Прогноз для 13-ї філії
X1_forecast = 1.2
X2_forecast = 15
X3_forecast = 10
X_forecast = np.array([[1, X1_forecast, X2_forecast, X3_forecast]])  # Включаємо константу

# Прогнозування
Y_forecast = model.predict(X_forecast)[0]

# Вивести прогноз
print(f"\nПрогноз для 13-ї філії: Y = {Y_forecast:.2f} млн. грн.")

# Побудова графіка
plt.figure(figsize=(10, 6))
plt.scatter(df["Філія"], df["Y"], color="blue", label="Дані")
plt.plot(df["Філія"], model.predict(X), color="red", label="Лінія тренду (регресія)")
plt.scatter(13, Y_forecast, color="green", marker="^", s=100, label="Прогноз для філії 13")
plt.xlabel("Номер філії")
plt.ylabel("Річний товарообіг (млн. грн.)")
plt.title("Модель множинної регресії та прогноз")
plt.legend()
plt.grid(True)
plt.show()
