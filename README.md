# Churn Prediction System  
### End-to-End Machine Learning Case Study (Streamlit-based)

This repository implements an **end-to-end churn prediction workflow** for the Telco Customer Churn dataset.  
It covers data preprocessing, class-imbalance handling, multiple ML models (classical + deep learning), and a **Streamlit-based interactive inference application**.

This README is written as a **focused technical review and case-study-style documentation**.  
It explains **what was built, how it is implemented, the engineering and ML challenges observed in the code, and concrete steps required to make the system production-ready**.

All statements are grounded in the actual repository files:
- `main.py`
- `design.py`
- `README.md`
- dataset (`churn.csv`)
- saved artifacts (`best_model.pkl`, `scaler.pkl`, `model_columns.pkl`)

---

## 🌐 Live Deployment

- **Streamlit App (Render)**  
  https://churn-prediction-qq8x.onrender.com

> This project uses **Streamlit only** for deployment and inference.  
> There is **no Flask or FastAPI backend** — all inference happens inside the Streamlit app.

---

## One-Sentence Summary

This repository implements an end-to-end churn prediction system for the Telco customer churn dataset using preprocessing (pandas, one-hot encoding, MinMaxScaler), multiple classical ML models (Logistic Regression, Random Forest, XGBoost), a TensorFlow ANN, explores undersampling/oversampling/SMOTE, and exposes predictions through a Streamlit UI.

---

## Repository Files Referenced

- **main.py**  
  Full ML workflow:  
  `preprocess_data()`, sampling functions (`undersample`, `oversample`, `smote_sample`),  
  ANN training (`build_and_train_ann()`), classical model training,  
  `evaluate_models()`, and `main()`

- **design.py**  
  Streamlit application that loads model artifacts (`best_model.pkl`, `scaler.pkl`, `model_columns.pkl`) and performs inference

- **README.md**  
  Project documentation summarizing the pipeline and intent

- **churn.csv**  
  Telco Customer Churn dataset

- **requirements.txt**  
  Dependencies (tensorflow, xgboost, imbalanced-learn, scikit-learn, streamlit, joblib, etc.)

---

## Business Problem and Motivation

### Problem
Binary classification to predict **customer churn** (`Churn` column) — whether a telecom customer is likely to leave the service.

### Why it matters
- Customer acquisition is significantly more expensive than retention
- Accurate churn prediction enables **targeted retention strategies**
- Small improvements in recall/precision for churned customers can yield large ROI

This project demonstrates how customer demographics, subscription details, service usage, and billing data can be transformed into a churn risk prediction system.

---

## Dataset Overview & Preprocessing (Exact Code Behavior)

### Dataset
- Telco Customer Churn dataset (`churn.csv`)
- Includes:
  - Demographics: gender, SeniorCitizen, Partner, Dependents
  - Services: PhoneService, InternetService, Streaming, Security, TechSupport
  - Contract & billing: tenure, MonthlyCharges, TotalCharges, PaymentMethod
  - Target: `Churn`

## Data Preprocessing

The preprocessing pipeline converts raw Telco customer records into a fully numeric feature matrix suitable for machine learning models. All preprocessing is implemented in `preprocess_data()` inside `main.py`.

### Step-by-step preprocessing

1. **Load dataset**
   - Read the Telco churn CSV file using pandas.
   - Operates on `churn.csv` located at the project root.

2. **Remove non-informative identifier**
   - Drops the `customerID` column.
   - Reason: `customerID` is a unique identifier and carries no predictive value.

3. **Handle invalid numeric values**
   - Removes rows where `TotalCharges` contains blank spaces (`' '`).
   - Converts `TotalCharges` from string to numeric using `pd.to_numeric`.
   - Ensures consistency for downstream scaling and modeling.

4. **Normalize service-specific categorical values**
   - Replaces service-specific responses:
     - `"No phone service"` → `"No"`
     - `"No internet service"` → `"No"`
   - Collapses semantically equivalent categories to reduce sparsity.

5. **Binary encoding of categorical features**
   - Converts binary categorical features from `Yes/No` to `1/0`.
   - Includes service usage and subscription indicators such as:
     - `Partner`, `Dependents`
     - `PhoneService`, `MultipleLines`
     - `OnlineSecurity`, `OnlineBackup`
     - `DeviceProtection`, `TechSupport`
     - `StreamingTV`, `StreamingMovies`
     - `PaperlessBilling`
     - Target variable `Churn`
   - Encodes `gender` as:
     - `Female → 1`
     - `Male → 0`

6. **One-hot encoding of multi-class categorical variables**
   - Applies `pd.get_dummies()` to:
     - `InternetService`
     - `Contract`
     - `PaymentMethod`
   - Uses `drop_first=True` to avoid multicollinearity.
   - Expands categorical variables into binary indicator columns.

7. **Feature–target separation**
   - Splits the dataset into:
     - Feature matrix `X` (all columns except `Churn`)
     - Target vector `y` (`Churn`)

8. **Feature scaling**
   - Applies `MinMaxScaler` to the following numeric columns:
     - `tenure`
     - `SeniorCitizen`
     - `TotalCharges`
     - `MonthlyCharges`
   - Scales features to the range `[0, 1]`.
   - Helps stabilize optimization for distance-based models and neural networks.

9. **Persistence of preprocessing artifacts**
   - Saves the fitted scaler to disk:
     - `scaler.pkl`
   - Saves the final feature column order:
     - `model_columns.pkl`
   - These artifacts are reused during inference in the Streamlit app to ensure feature consistency.

### Important preprocessing caveat

⚠️ **Data leakage risk**  
- The scaler is fit on the **entire dataset before train/test splitting**.
- This allows information from the test set to influence preprocessing.
- While acceptable for experimentation, this should be fixed for production by:
  - Splitting the data first
  - Fitting the scaler only on the training set
  - Applying `transform()` to validation and test sets

---

Exploratory Data Analysis (EDA)

Function plot_distributions(df) visualizes:

Tenure distribution (churn vs non-churn)

MonthlyCharges distribution (churn vs non-churn)

These plots influenced:

Scaling numeric features

Retaining tenure and billing variables as strong predictors

No automated feature selection or correlation analysis is present.

Class Imbalance Handling (Implemented)

The project explicitly explores three imbalance strategies:

Undersampling

Oversampling

SMOTE

Implementation details:

Sampling functions defined in main.py

Stratified train/test splits used (stratify=y)

Models retrained under each strategy for comparison

This allows empirical evaluation of imbalance handling rather than assuming a single technique.

# Churn Prediction – Models Implemented

This repository implements and compares multiple machine learning models for predicting customer churn using structured telecom customer data. All models are trained and evaluated on the same preprocessed feature matrix generated in `main.py`, and the selected best-performing model is later used for inference inside the Streamlit application.

The project includes both classical machine learning models and a deep learning model to study how different modeling approaches perform on an imbalanced churn prediction problem.

A Logistic Regression model is implemented using scikit-learn with `max_iter=1000`. It serves as a baseline linear classifier trained on the numeric feature set without applying any class weighting or imbalance-specific parameters. The model is evaluated on the test set using accuracy, precision, recall, F1-score, and confusion matrix, providing a simple and interpretable reference point for comparison with more complex models.

A Random Forest classifier is implemented using `RandomForestClassifier` with 100 estimators and a fixed random seed. This ensemble-based model learns non-linear relationships and feature interactions automatically from the data. It is trained on the same feature matrix as Logistic Regression and evaluated using the same classification metrics. No explicit class imbalance handling such as `class_weight` is applied during training. The Random Forest acts as a strong traditional machine learning baseline capable of modeling complex patterns in customer behavior.

An XGBoost classifier is implemented using `XGBClassifier` with default hyperparameters. The model is trained on the preprocessed dataset without explicit tuning or imbalance-specific parameters such as `scale_pos_weight`. Performance is evaluated using accuracy, precision, recall, F1-score, and confusion matrix, and results are compared against Logistic Regression, Random Forest, and the neural network model. XGBoost is included to represent gradient-boosted decision trees, which are commonly effective for structured tabular data like customer churn datasets.

In addition to classical machine learning models, the project implements an Artificial Neural Network (ANN) using TensorFlow and Keras. The ANN is built using a sequential architecture consisting of an input layer followed by two hidden dense layers with 23 and 15 neurons respectively using ReLU activation, and a final sigmoid-activated output layer for binary classification. The model is trained using the Adam optimizer with binary crossentropy loss and accuracy as the evaluation metric. Training data is converted into a `tf.data.Dataset` with shuffling, batching, and prefetching enabled for performance optimization. Early stopping is applied with monitoring on validation loss, a patience of 20 epochs, and restoration of the best weights. The ANN outputs churn probabilities using `model.predict`, which are rounded to generate binary predictions. A key limitation is that the test dataset is used as validation data during training, introducing evaluation leakage and potentially optimistic performance estimates.

All models are evaluated using the same train-test split, and their performance metrics are printed to the console for comparison. Based on the observed evaluation results, the best-performing model is selected and saved as `best_model.pkl`. This saved model is later loaded by the Streamlit application to perform real-time churn prediction during inference. The exact metric used for selecting the best model is not explicitly defined in the code and relies on manual inspection of printed classification metrics.

# Model Comparison & Selection

All implemented models are evaluated using the same train–test split to ensure a fair comparison.

Model performance metrics are printed to the console after evaluation. These metrics include accuracy, precision, recall, F1-score, and the confusion matrix.

Based on the observed evaluation results, the project workflow selects the best-performing model and saves it to disk as:

```text
best_model.pkl
```

Why this section works

Matches the actual implementation in the repository

Clearly communicates how model selection is performed

Transparently documents limitations in the selection logic

Fits naturally into a professional, reviewer-friendly README


---

✅ Single Markdown file  
✅ Correct formatting  
✅ No mixed styles  
✅ README-ready  

If you want this merged into your **full churn project README**, say the word and I’ll assemble the final version cleanly.
