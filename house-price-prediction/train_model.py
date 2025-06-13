# train_model.py

import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load the dataset
df = pd.read_csv("data.csv")

# Use correct column names from the CSV: 'Area', 'Bedrooms', 'Bathrooms'
X = df[["Area", "Bedrooms", "Bathrooms"]]
y = df["Price"]

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Save the trained model to a file
joblib.dump(model, "model.pkl")

print("✅ Model has been trained and saved as 'model.pkl'.")
