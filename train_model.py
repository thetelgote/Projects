import pandas as pd
import os
from sklearn.linear_model import LinearRegression
import joblib

print(f"📂 Current Directory: {os.getcwd()}")
print(f"📄 data.csv exists? {os.path.exists('data.csv')}")
print("📑 Reading file content...")

with open("data.csv", "r") as f:
    content = f.read()
    print("--------")
    print(content)
    print("--------")

df = pd.read_csv("data.csv")
print("✅ DataFrame Loaded Successfully!")
print(df.head())

x = df[["area", "bedrooms", "bathrooms"]]
y = df["price"]

model = LinearRegression()
model.fit(x, y)

joblib.dump(model, "model.pkl")
print("✅ Model trained and saved successfully.")
