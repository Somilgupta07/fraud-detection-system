# 💳 Fraud Detection System using Machine Learning

A Machine Learning project that detects fraudulent credit card transactions using multiple ML models and deploys the prediction interface with **Streamlit**.

The system analyzes transaction features such as transaction amount, location, category, and merchant information to predict whether a transaction is **fraudulent or legitimate**.

---

## 🚀 Live Demo

After deployment the app will be available at:

Streamlit App Link

---

## 🧠 Machine Learning Pipeline

This project follows a complete **ML Engineer workflow**:

1. Data Collection
2. Data Cleaning
3. Feature Engineering
4. Encoding Categorical Variables
5. Handling Imbalanced Dataset
6. Model Training
7. Model Evaluation
8. Model Selection
9. Model Deployment with Streamlit

---

## 📊 Dataset

The dataset contains credit card transaction details.

Files used:

fraudTrain.csv
fraudTest.csv

### Dataset Size

| Dataset | Rows      | Columns |
| ------- | --------- | ------- |
| Train   | 1,296,675 | 23      |
| Test    | 555,719   | 23      |

---

## ⚙️ Features Used in Model

Some important features used:

- Transaction Amount (`amt`)
- Transaction Category (`category`)
- Customer Latitude (`lat`)
- Customer Longitude (`long`)
- Merchant Latitude (`merch_lat`)
- Merchant Longitude (`merch_long`)

Target Variable:

is_fraud

0 → Legitimate Transaction  
1 → Fraudulent Transaction

---

## 🤖 Models Used

Three machine learning models were trained and evaluated.

### 1️⃣ Logistic Regression

Baseline model for classification.

### 2️⃣ Random Forest

Ensemble tree model that improves accuracy.

### 3️⃣ XGBoost (Final Model)

Gradient boosting model used for best performance.

XGBoost was selected as the **final deployed model**.

---

## 📈 Model Evaluation Metrics

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Fraud datasets are **highly imbalanced**, so **Recall and F1 Score** were important metrics.

---

## 🖥️ Streamlit Web Application

The trained model is deployed with **Streamlit** to allow users to test transactions interactively.

Users can input:

- Transaction Amount
- Category
- Customer Location
- Merchant Location

The model then predicts whether the transaction is:
Legitimate
or
Fraudulent

---

## 📸 Application Screenshot

(Add screenshot here)

---

## 📁 Project Structure

fraud-detection-project
│
├── fraudTrain.csv
├── fraudTest.csv
├── fraud_model.pkl
├── encoder.pkl
├── model_columns.pkl
│
├── model.ipynb
├── streamlit_app.py
│
├── requirements.txt
├── README.md

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Streamlit
- Joblib

---

## ▶️ Run Locally

### 1️⃣ Clone Repository

git clone https://github.com/Somilgupta07/fraud-detection-system.git

### 2️⃣ Install Dependencies

pip install -r requirements.txt

### 3️⃣ Run Streamlit App

streamlit run streamlit_app.py

---

## 🌐 Deployment

The project can be deployed using:

- Streamlit Cloud
- Render
- HuggingFace Spaces

Recommended:
Streamlit Cloud

---

## 🎯 Future Improvements

- Real-time transaction simulation
- Feature importance visualization
- Fraud probability score
- Dashboard with analytics

---

## 👨‍💻 Author

Somil Gupta

Machine Learning & Full Stack Developer

---

## ⭐ If you like this project

Please give it a **star ⭐ on GitHub**.
