import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib

print("🚀 Starting the model training process...")

file_path = 'F:/1 PROJECT/AI ML/BRAIN/Housing.csv'
try:
    df = pd.read_csv(file_path)
    print(f"✅ Loaded dataset successfully from '{file_path}'")
except FileNotFoundError:
    print(f"🛑 Error: '{file_path}' not found. Make sure the file exists in the correct folder.")
    exit()


binary_cols = ['mainroad', 'guestroom', 'basement', 'hot_water', 'air_conditioner', 'prefarea']
for col in binary_cols:
    df[col] = df[col].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)


df = pd.get_dummies(df, columns=['furnishing_status'], drop_first=True, dtype=int)

df['area_per_bedroom'] = df['area'] / (df['bedroom'] + 0.01) 

X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Data split: {X_train.shape[0]} training rows, {X_test.shape[0]} testing rows")

numerical_cols = ['area', 'bedroom', 'bathroom', 'stores', 'parking', 'area_per_bedroom']
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
print("✅ Numerical features scaled successfully.")

print("⚡ Training the Linear Regression model...")
model = LinearRegression()
model.fit(X_train, y_train)
print("✅ Model training complete.")

r2_score_test = model.score(X_test, y_test)
print(f"📊 R-squared score on test set: {r2_score_test:.4f}")

joblib.dump(model, 'housing_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
print("\n🎉 Success! Model and scaler have been saved as:")
print("-> 🧠 housing_model.joblib")
print("-> 📏 scaler.joblib")