🏡 House Price Prediction using Machine Learning
📊 Predict housing prices based on features like area, bedrooms, bathrooms, parking, and furnishing status using Linear Regression in Python.
🚀 Project Overview

This project applies supervised machine learning to predict house prices based on a dataset (Housing.csv) that includes structural and location-related features.
The workflow includes:

Data preprocessing and encoding

Feature scaling using StandardScaler

Model training with Linear Regression

Interactive prediction through user input

Model and scaler saving using Joblib

🧠 Model Training
🔹 File: Train_Housing_Model.py

This script:

Loads the dataset

Encodes categorical data (Yes/No, Furnishing status)

Adds a new feature: area_per_bedroom

Scales numeric features

Trains a Linear Regression model

Saves the trained model and scaler

✅ Outputs:

housing_model.joblib

scaler.joblib

⚙️ Safe Training Version
🔹 File: Train_Housing_Model_Safe.py

Same as above, but includes:

Try/except blocks for missing files

Detailed progress messages

Better for debugging and stable environments

💬 Predicting Prices
🔹 File: Predict_House_Price.py

This script:

Loads your saved model, scaler, and dataset

Asks the user for input (like area, bedrooms, etc.)

Automatically fills missing data with median values

Applies correct feature scaling

Predicts and displays the estimated price

📘 Example Run

🚀 Starting house price prediction...
✅ Model, scaler, and original dataset loaded successfully.

--- 🏡 Enter House Details ---
Enter the area in sq ft (e.g., 3500): 2888
Enter the number of bedrooms (e.g., 3): 6
Enter the number of bathrooms (e.g., 2): 2.5
...
The estimated price of the house is: $95,320.55

🧩 Folder Structure
F:/1 PROJECT/AI ML/BRAIN/
│
├── Housing.csv
├── Train_Housing_Model.py
├── Train_Housing_Model_Safe.py
├── Predict_House_Price.py
├── housing_model.joblib
└── scaler.joblib

📦 Requirements

Create a requirements.txt file with this content:

pandas
numpy
scikit-learn
joblib


Install with:

pip install -r requirements.txt

🧰 Tools & Libraries Used

Python 3.10+

Pandas for data analysis

NumPy for numerical operations

Scikit-learn for ML model and scaling

Joblib for saving model and scaler

📈 Model Performance

The Linear Regression model achieved an R² score on test data:

R² Score: 0.78 (approx.)


(Varies slightly based on dataset split.)

🌟 Features

✅ Clean and modular Python scripts
✅ Handles missing user inputs
✅ Scalable and easily extendable
✅ Console-based interactive prediction
✅ Uses real-world housing data


💡 Future Improvements

Add GUI (Tkinter / Streamlit)

Include more ML models (Random Forest, XGBoost)

Deploy via Flask or FastAPI

👨‍💻 Author

Relton Abishek Jayaraj
📍 Boston, MA
🎓 M.S. in Business Analytics @ UMass Boston
🎵 Worship leader & data enthusiast

🔗 GitHub: Rel10Jayaraj

💼 LinkedIn: linkedin.com/in/reltonabishek
