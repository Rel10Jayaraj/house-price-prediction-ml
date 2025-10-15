import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib 

print("🚀 Starting model training...")

df = pd.read_csv(r'F:/1 PROJECT/AI ML/BRAIN/Housing.csv')


binary_cols = ['mainroad', 'guestroom', 'basement', 'hot_water', 'air_conditioner', 'prefarea']
for col in binary_cols:
    df[col] = df[col].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)


df = pd.get_dummies(df, columns=['furnishing_status'], drop_first=True, dtype=int)


df['area_per_bedroom'] = df['area'] / (df['bedroom'] + 0.01) 


X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


numerical_cols = ['area', 'bedroom', 'bathroom', 'stores', 'parking', 'area_per_bedroom']
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R-squared score with 'area_per_bedroom': {r2:.4f}")


joblib.dump(model, 'housing_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
print("\n✅ Success! Model and scaler have been saved.")