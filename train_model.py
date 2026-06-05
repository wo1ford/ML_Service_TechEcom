"""Скрипт обучения модели / прогноза дефицита курьеров ( взял с Kaggle соревнования ecom )"""
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score

print(" =#= Обучение модели прогноза дефицита курьеро =#= ")

np.random.seed(42)
n = 10000

# Генерация данных
data = pd.DataFrame({
    "orders": np.random.randint(5, 500, n),
    "couriers": np.random.randint(1, 50, n),
    "hour": np.random.randint(0, 24, n),
    "day_of_week": np.random.randint(0, 7, n),
    "city_zone": np.random.choice(["center", "north", "south", "east", "west"], n),
    "weather": np.random.choice(["clear", "rain", "snow", "fog"], n, p=[0.4, 0.3, 0.2, 0.1]),
    "is_holiday": np.random.choice([0, 1], n, p=[0.85, 0.15]),
    "avg_distance_km": np.round(np.random.uniform(0.5, 25, n), 1),
    "active_orders_ratio": np.round(np.random.uniform(0.3, 1.0, n), 2),
})

# Целевая перемен.
base_load = data["orders"] / (data["couriers"] + 0.1)
hour_penalty = np.where((data["hour"] >= 11) & (data["hour"] <= 14), 0.3,
                 np.where((data["hour"] >= 18) & (data["hour"] <= 21), 0.4, 0.0))
weather_penalty = data["weather"].map({"clear": 0, "rain": 0.2, "snow": 0.5, "fog": 0.3})
weekend_penalty = np.where(data["day_of_week"] >= 5, 0.15, 0)
holiday_penalty = data["is_holiday"] * 0.5
distance_penalty = data["avg_distance_km"] / 50

deficit_raw = (
    base_load / 5 + hour_penalty * base_load + weather_penalty * base_load
    + weekend_penalty * base_load + holiday_penalty * base_load
    + distance_penalty * base_load - data["couriers"] * 0.5
    + np.random.normal(0, 0.5, n)
)
data["deficit"] = np.maximum(0, np.ceil(deficit_raw)).astype(int)

# Фичи
data["orders_per_courier"] = data["orders"] / (data["couriers"] + 1)
data["load_index"] = data["orders_per_courier"] * data["active_orders_ratio"]
data["is_peak"] = ((data["hour"] >= 11) & (data["hour"] <= 14) | (data["hour"] >= 18) & (data["hour"] <= 21)).astype(int)
data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
data["rush_factor"] = data["is_peak"] * data["orders_per_courier"]
data["weather_severity"] = data["weather"].map({"clear": 0, "rain": 1, "fog": 2, "snow": 3})

features = [
    "orders", "couriers", "hour_sin", "hour_cos",
    "is_peak", "is_weekend", "is_holiday",
    "city_zone", "weather_severity",
    "avg_distance_km", "active_orders_ratio",
    "orders_per_courier", "load_index", "rush_factor"
]

X = data[features]
y = data["deficit"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first", sparse_output=False), ["city_zone"])
], remainder="passthrough")

model = Pipeline([
    ("preprocessor", preprocessor),
    ("scaler", StandardScaler()),
    ("regressor", GradientBoostingRegressor(
        n_estimators=150, max_depth=5, learning_rate=0.1,
        subsample=0.8, random_state=42
    ))
])

model.fit(X_train, y_train)

y_pred = np.maximum(0, np.round(model.predict(X_test))).astype(int)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
cv_scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")

print(f"\nMAE: {mae:.2f} | R²: {r2:.4f} | CV MAE: {-cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

with open("Model/model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("Model/features.pkl", "wb") as f:
    pickle.dump(features, f)

print("Модель сохранена: Model/model.pkl")