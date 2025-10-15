import joblib
import pandas as pd
import numpy as np

print("🚀 Starting house price prediction...")

try:
    model = joblib.load('housing_model.joblib')
    scaler = joblib.load('scaler.joblib')
    df_original = pd.read_csv('Housing.csv')
    print("✅ Model, scaler, and original dataset loaded successfully.")
except FileNotFoundError as e:
    print(f"🛑 Error: Could not find a required file: {e.filename}.")
    exit()

def get_user_input(prompt, column_name):
    """Get numerical input or use median if 'N/A'."""
    while True:
        user_input = input(prompt).strip().lower()
        if user_input == 'n/a':
            median_value = df_original[column_name].median()
            print(f"-> Using median value for '{column_name}': {median_value}")
            return median_value
        try:
            return float(user_input)
        except ValueError:
            print("❌ Invalid input. Enter a number or 'N/A'.")

def get_yes_no_input(prompt):
    """Get 'yes'/'no' input and convert to 1/0."""
    while True:
        user_input = input(prompt).strip().lower()
        if user_input == 'yes':
            return 1
        elif user_input == 'no':
            return 0
        elif user_input == 'n/a':
            print("-> Defaulting to 'no' (0).")
            return 0
        else:
            print("❌ Invalid input. Enter 'yes', 'no', or 'N/A'.")

print("\n--- 🏡 Enter House Details (type 'N/A' if unknown) ---")
new_house = {
    'area': get_user_input("Enter the area in sq ft (e.g., 3500): ", 'area'),
    'bedroom': get_user_input("Enter the number of bedrooms (e.g., 3): ", 'bedroom'),
    'bathroom': get_user_input("Enter the number of bathrooms (e.g., 2): ", 'bathroom'),
    'stores': get_user_input("Enter the number of floors/stories (e.g., 2): ", 'stores'),
    'parking': get_user_input("Enter the number of parking spots (e.g., 2): ", 'parking'),
    'mainroad': get_yes_no_input("Is the house on a main road? (yes/no): "),
    'guestroom': get_yes_no_input("Does it have a guest room? (yes/no): "),
    'basement': get_yes_no_input("Does it have a basement? (yes/no): "),
    'hot_water': get_yes_no_input("Does it have hot water heating? (yes/no): "),
    'air_conditioner': get_yes_no_input("Does it have air conditioning? (yes/no): "),
    'prefarea': get_yes_no_input("Is it in a preferred area? (yes/no): "),
}

while True:
    furnishing = input("What is the furnishing status? (furnished/semi-furnished/unfurnished/n/a): ").strip().lower()
    if furnishing == 'semi-furnished':
        new_house['furnishing_status_Semi-furnished'] = 1
        new_house['furnishing_status_Unfurnished'] = 0
        break
    elif furnishing == 'unfurnished':
        new_house['furnishing_status_Semi-furnished'] = 0
        new_house['furnishing_status_Unfurnished'] = 1
        break
    else:  
        new_house['furnishing_status_Semi-furnished'] = 0
        new_house['furnishing_status_Unfurnished'] = 0
        if furnishing == 'n/a':
            print("-> Defaulting to 'furnished'.")
        break

input_df = pd.DataFrame([new_house])
input_df['area_per_bedroom'] = input_df['area'] / (input_df['bedroom'] + 0.01)

training_columns = [
    'area', 'bedroom', 'bathroom', 'stores', 'mainroad', 'guestroom',
    'basement', 'hot_water', 'air_conditioner', 'parking', 'prefarea',
    'furnishing_status_Semi-furnished', 'furnishing_status_Unfurnished',
    'area_per_bedroom'
]
input_df = input_df.reindex(columns=training_columns, fill_value=0)

numerical_cols = ['area', 'bedroom', 'bathroom', 'stores', 'parking', 'area_per_bedroom']
input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

if hasattr(model, "feature_names_in_"):
    input_df = input_df[model.feature_names_in_]
else:
    print("⚠️ Model does not store feature names; using current column order.")

predicted_price = model.predict(input_df)

print("\n--- 🏡 House Price Prediction ---")
print(f"The estimated price of the house is: ${predicted_price[0]:,.2f}")