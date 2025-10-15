# 🏡 House Price Prediction for Massachusetts  M.S. in Business Analytics 

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Libraries](https://img.shields.io/badge/Libraries-Pandas%20%7C%20Scikit--learn-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

A machine learning project that predicts housing prices in Massachusetts using a Linear Regression model. The application is built with Python and Scikit-learn, allowing for interactive price estimation based on user-provided house features.

---

## 🚀 Project Overview

This project demonstrates a complete machine learning workflow for a regression task. It takes a dataset of housing information, processes it, trains a predictive model, and provides an interactive console application for users to get instant price estimates.

### ✨ Key Features
-   ✅ **Clean & Modular Code:** Scripts are separated by concern (training vs. prediction).
-   🤖 **Interactive Prediction:** A user-friendly command-line interface to input house features.
-   🛠️ **Data Preprocessing:** Handles categorical data, feature engineering, and scaling.
-   📦 **Saved Models:** Uses `Joblib` to save and reuse the trained model and scaler, so you don't have to retrain every time.
-   📉 **Performance Metrics:** Evaluates the model using the R² score to measure its accuracy.

---

## 📊 Model Performance

The Linear Regression model was trained and evaluated to determine its predictive accuracy on unseen data.

> **R² Score (Test Data): `~0.78`**

*This score indicates that the model can explain approximately 78% of the variance in the house prices, which is a solid result for a baseline model. The score may vary slightly with different data splits.*

---

## ⚙️ Project Workflow & Files

The repository is structured into two main Python scripts for training the model and making predictions.

### 🧠 1. Model Training
🔹 **File:** `Train_Housing_Model.py`

This script is the core of the training process. It performs the following steps:
1.  **Loads** the `Housing.csv` dataset.
2.  **Encodes** categorical columns (`yes`/`no` and furnishing status) into numerical format.
3.  **Engineers** a new feature (`area_per_bedroom`) to improve model performance.
4.  **Splits** the data into training and testing sets.
5.  **Scales** numerical features using `StandardScaler` to normalize the data.
6.  **Trains** a Linear Regression model on the prepared data.
7.  **Saves** the final model (`housing_model.joblib`) and the scaler (`scaler.joblib`).

*A safe version, `Train_Housing_Model_Safe.py`, is also included with enhanced error handling and logging.*

### 💬 2. Price Prediction
🔹 **File:** `Predict_House_Price.py`

This script provides an interactive interface for the user:
1.  **Loads** the pre-trained `housing_model.joblib` and `scaler.joblib`.
2.  **Prompts** the user to enter details for a house (area, bedrooms, bathrooms, etc.).
3.  **Handles Missing Input** by automatically using the median value from the original dataset.
4.  **Prepares** the input data by scaling and ordering features exactly like the training data.
5.  **Predicts** the price and displays the final, formatted result to the user.

---

## 🛠️ Getting Started

Follow these steps to run the project on your local machine.

### 📦 Requirements
First, create a `requirements.txt` file with the following content:
pandas numpy scikit-learn joblib


### ⚙️ Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-folder>
    ```

2.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Train the model (only needs to be done once):**
    ```bash
    python Train_Housing_Model.py
    ```
    *This will create `housing_model.joblib` and `scaler.joblib`.*

4.  **Run the prediction script:**
    ```bash
    python Predict_House_Price.py
    ```

### 📘 Example Run
🚀 Starting house price prediction... ✅ Model, scaler, and original dataset loaded successfully.

--- 🏡 Enter House Details (type 'N/A' if unknown) --- Enter the area in sq ft (e.g., 3500): 2888 
Enter the number of bedrooms (e.g., 3): 6 
Enter the number of bathrooms (e.g., 2): 2.5 
Enter the number of stories/floors (e.g., 2): 3 
Enter the number of parking spots (e.g., 2): 1 
Is the house on a main road? (yes/no): no 
Does it have a guest room? (yes/no): yes 
Does it have a basement? (yes/no): yes 
Does it have hot water heating? (yes/no): yes 
Does it have air conditioning? (yes/no): yes 
Is it in a preferred area? (yes/no): yes 
What is the furnishing status? (furnished/semi-furnished/unfurnished/n/a): unfurnished

--- 🏡 House Price Prediction --- The estimated price of the house is: $95,320.55

---

## 💡 Future Improvements
-   **Add a GUI:** Develop a graphical interface using `Tkinter` or `Streamlit` for a more user-friendly experience.
-   **Experiment with Models:** Incorporate more advanced models like Random Forest, Gradient Boosting, or XGBoost.
-   **Deploy as an API:** Wrap the prediction script in an API using `Flask` or `FastAPI` to serve the model online.

---

## 👨‍💻 Author

**Relton Abishek Jayaraj**

📍 Boston, MA
<br>
🎓 M.S. in Business Analytics @ UMass Boston
<br>
Data/Business Analyst

<a href="https://github.com/Rel10Jayaraj" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"/>
</a>
<a href="https://linkedin.com/in/reltonabishekjayaraj" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"/>
</a>
