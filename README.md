# 📊 Customer Churn Prediction & Analysis

A comprehensive machine learning project designed to predict customer churn in telecommunications. This solution allows businesses to identify customers at risk of leaving and take proactive retention measures. It features a complete pipeline from raw data processing to a deployed web interface.

## 🌐 Live Demo
Check out the deployed application on Render:  
👉 **[Churn Prediction App](https://churn-prediction-qq8x.onrender.com)**  

> [!NOTE]  
> Please wait approximately **50 seconds** for the application to load, as the server may spin down during inactivity.

## 📂 Project Structure

| File | Description |
|------|-------------|
| `main.py` | The core orchestration script. Handles data loading, preprocessing, EDA, model training (ANN & Classical), resampling experiments, evaluation, and saving the optimal model. |
| `design.py` | A generic Streamlit web application that serves the trained model (`best_model.pkl`), providing an interactive UI for real-time predictions. |
| `churn.csv` | The source dataset containing customer demographics, services, and billing information. |
| `best_model.pkl` | The serialized machine learning model (selected automatically based on highest accuracy) used by the web app. |
| `requirements.txt` | Lists all Python dependencies required to run the project. |
| `main.ipynb` | Jupyter Notebook environment for experimental analysis and visualization. |

---

## ⚙️ Technical Architecture

### 1. Data Preprocessing Pipeline
The `preprocess_data` function in `main.py` performs rigorous data cleaning and transformation:
- **Data Cleaning**:
  - Drops irrelevant columns (`customerID`).
  - Removes rows with empty `TotalCharges` strings.
  - Converts `TotalCharges` to numeric format.
- **Label Normalization**:
  - Standardizes text: "No phone service" → "No", "No internet service" → "No".
- **Encoding**:
  - **Binary Encoding** (0/1): `Gender`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, `Churn`, etc.
  - **One-Hot Encoding**: Used for multi-class categorical features (`InternetService`, `Contract`, `PaymentMethod`).
- **Feature Scaling**:
  - Applies **MinMaxScaler** to normalize continuous variables: `tenure`, `SeniorCitizen`, `TotalCharges`, `MonthlyCharges`.

### 2. Exploratory Data Analysis (EDA)
The system visualizes churn trends using Matplotlib:
- Distribution of **Tenure** for Churn vs. Non-Churn customers.
- Distribution of **Monthly Charges** impacting churn rates.

### 3. Handling Class Imbalance
To address the natural imbalance in churn datasets, the comprehensive `main.py` implements three strategies:
- **Undersampling**: Reduces the majority class (Non-Churn) to match the minority class count.
- **Oversampling**: Randomly duplicates minority class (Churn) samples.
- **SMOTE (Synthetic Minority Over-sampling Technique)**: Generates synthetic instances for the minority class to create a balanced dataset.

---

## 🧠 Machine Learning Models

The project implements and compares four distinct model architectures:

### 1. Artificial Neural Network (ANN)
Built with TensorFlow/Keras:
- **Architecture**:
  - Input Layer: Matches feature shape.
  - Hidden Layer 1: **23 neurons**, ReLU activation.
  - Hidden Layer 2: **15 neurons**, ReLU activation.
  - Output Layer: **1 neuron**, Sigmoid activation (probability 0-1).
- **Training Config**:
  - Optimizer: **Adam**.
  - Loss Function: **Binary Crossentropy**.
  - **Early Stopping**: Monitors `val_loss` with a patience of 20 epochs to prevent overfitting.
  - Max Epochs: 2000 (with early stopping).

### 2. XGBoost Classifier
A gradient boosting framework:
- **Estimators**: 200 trees.
- **Learning Rate**: 0.1.
- **Max Depth**: 6.
- **Subsample/Colsample**: 0.8 (reduces overfitting).

### 3. Random Forest Classifier
Ensemble learning method:
- **Estimators**: 100 decision trees.
- **Random State**: Fixed for reproducibility.

### 4. Logistic Regression
Baseline linear classifier for binary classification comparisons.

---

## 📊 Evaluation & Metrics

The `evaluate_models` function generates a comparative report for all trained models based on:
- **Accuracy**: Overall correctness.
- **Precision**: Accuracy of positive predictions (Churn).
- **Recall**: Ability to capture actual churn cases.
- **F1-Score**: Harmonic mean of precision and recall.

*The script automatically identifies the model with the highest Accuracy and saves it as `best_model.pkl`.*

---

## 🖥️ Web Application (`design.py`)

The project includes a **Streamlit** dashboard for end-user interaction.

### Features
- **User Interface**: Split-layout design separating Customer Profile (Demographics) from Service/Billing details.
- **Real-time Processing**:
  - Accepts raw inputs (e.g., "Yes/No", "Fiber optic", numeric charges).
  - Internally scales inputs using a `MinMaxScaler` fitted to the training range assumptions (`Tenure`: 0-72, `Monthly`: 0-200, `Total`: 0-10000).
- **Visual Feedback**:
  - **High Risk**: Displays a warning card with a gradient red background.
  - **Low Risk**: Displays a success card with a gradient blue background.
  - Shows exact probability percentages for both outcomes.

---

## � Getting Started

### Prerequisites
- Python 3.8+
- pip packet manager

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/mayank200604/Churn_prediction.git
   ```
2. Navigate to the directory:
   ```bash
   cd Churn_prediction
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Training Pipeline
To process data, train models, and generate the best model file:
```bash
python main.py
```

### Launching the App
To start the prediction interface:
```bash
streamlit run design.py
```
Access the tool at `http://localhost:8501`.

---

## 📝 Author
**Mayank** - [GitHub Profile](https://github.com/mayank200604)
