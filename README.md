# 📊 Customer Churn Prediction — End-to-End ML + ANN Project

This project predicts **customer churn** using multiple machine learning techniques including **Artificial Neural Networks (ANN)**, **Logistic Regression**, **Random Forest**, and **XGBoost**.  
It covers the full ML lifecycle — **data preprocessing, imbalance handling, model comparison, best model export**, and a **Streamlit dashboard** for real-time prediction.

---

## 🗂️ Project Structure

├── design.py # Streamlit app for churn prediction using best trained model
├── main.py # Full workflow: preprocessing, sampling, modeling & evaluation
├── main.ipynb # Notebook version of preprocessing & model experimentation
├── churn.csv # Telco customer churn dataset
├── requirements.txt # Project dependencies
└── README.md # Documentation


---

## 📌 Dataset Information

The dataset contains **customer demographics, billing info, service subscriptions, and churn labels**.  
Target column: **Churn (Yes=1 / No=0)**

Key columns used:
- `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- `PhoneService`, `InternetService`, `Contract`, `PaymentMethod`
- `MonthlyCharges`, `TotalCharges`, `tenure`, etc.

---

## 🔧 Data Preprocessing

All preprocessing is handled inside `preprocess_data()`:

- Drop `customerID`
- Remove blank `TotalCharges` values, then convert to numeric
- Replace special text values (`No phone service`, `No internet service`)
- Encode binary labels (`Yes` / `No` → `1` / `0`)
- One-hot encode categorical columns:
  - `InternetService`, `Contract`, `PaymentMethod`
- Normalize numerical columns using **MinMaxScaler**:
  - `tenure`, `SeniorCitizen`, `TotalCharges`, `MonthlyCharges`

Output:
- `X` — processed features
- `y` — churn labels
- `df` — clean dataframe

---

## 📉 Class Imbalance Handling

The dataset has more non-churn customers than churn customers.  
Three balancing strategies are implemented:

| Method | Description | Usage |
|--------|------------|-------|
| **Undersampling** | Reduce majority class by sampling | `undersample(df)` |
| **Oversampling** | Duplicate minority class examples | `oversample(df)` |
| **SMOTE** | Generate synthetic minority samples | `smote_sample(X, y)` |

Each method is evaluated using ANN during training.

---

## 🤖 Machine Learning Models

Implemented models:

| Model | Function | Notes |
|-------|----------|-------|
| Logistic Regression | `train_logistic_regression()` | Baseline |
| Random Forest | `train_random_forest()` | Ensemble |
| XGBoost | `train_xgboost()` | Boosting |
| ANN (TensorFlow) | `build_and_train_ann()` | Deep classifier |

Metrics printed for each model:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

---

## 🧠 ANN Architecture

The neural network used:

| Layer | Units | Activation |
|-------|-------|------------|
| Dense | 23 | ReLU |
| Dense | 15 | ReLU |
| Dense | 1 | Sigmoid |

Additional features:
- Early stopping for overfitting control
- TensorFlow `Dataset` API for batching & prefetching
- Can handle weighted or unweighted training

---

## 🥇 Model Selection & Saving

`evaluate_models()`:
- Trains all models
- Compares accuracy, precision, recall, F1
- Selects highest performance model
- Saves it as `best_model.pkl`.
  
  This file is later used by the Streamlit app.

---

## 🌐 Streamlit App (`design.py`)

The app provides a **GUI to predict churn probability** based on manual inputs.

User selects:
- Customer profile: `SeniorCitizen`, `Partner`
- Service details: `InternetService`, `Contract`
- Billing: `Tenure`, `MonthlyCharges`, `TotalCharges`
- Support services: `OnlineSecurity`, `TechSupport`

Features:
- Encodes & scales inputs internally
- Predicts using `best_model.pkl`
- Displays risk:
  - **⚠️ HIGH RISK**
  - **✅ LOW RISK**
- Shows churn probability percent

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mayank200604/Churn_prediction.git
   cd Churn_prediction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   *Key dependencies: `tensorflow`, `xgboost`, `streamlit`, `imbalanced-learn`, `scikit-learn`*

## 🚀 Usage

### 1. Train the Model
Run the training script to generate the model and necessary artifacts (`scaler.pkl`, `model_columns.pkl`):
```bash
python main.py
```

### 2. Run the Web App
Launch the Streamlit dashboard:
```bash
streamlit run design.py
```

## 📊 Visualizations

`plot_distributions(df)` shows:
- Tenure distribution for churn vs non-churn
- Monthly charges distribution for churn vs non-churn

## 🔮 Future Enhancements

- Hyperparameter tuning for ANN & XGBoost
- Add ROC-AUC, SHAP explainability
- Export scaler + preprocessing pipeline
- API deployment (FastAPI / Flask)
- Cloud deployment (Streamlit Cloud / Render)

## 👤 Author
**Mayank** — B.Sc CS | ML & AI Enthusiast  
Actively building ML & Deep Learning projects.