# ================== IMPORTS ==================
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib




# ================== DATA PREPROCESSING ==================
def preprocess_data(filepath="churn.csv"):
    """Load and preprocess churn dataset."""
    df = pd.read_csv(filepath)
    df.drop('customerID', axis=1, inplace=True)

    # Remove blank TotalCharges rows and convert to numeric
    df = df[df.TotalCharges != ' '].copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

    # Replace text labels
    df.replace('No phone service', 'No', inplace=True)
    df.replace('No internet service', 'No', inplace=True)
    df['gender'].replace({'Female': 1, 'Male': 0}, inplace=True)

    # One-hot encoding for categorical columns
    categorical_columns = ['InternetService', 'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # Encode Yes/No columns
    yes_no_columns = [
        'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies',
        'PaperlessBilling', 'Churn'
    ]
    for col in yes_no_columns:
        df[col].replace({'Yes': 1, 'No': 0}, inplace=True)

    # Features and labels
    X = df.drop(['Churn'], axis=1)
    y = df['Churn']

    # Scale numeric columns
    scaler = MinMaxScaler()
    columns_to_scale = ['tenure', 'SeniorCitizen', 'TotalCharges', 'MonthlyCharges']
    X[columns_to_scale] = scaler.fit_transform(X[columns_to_scale])

    # Save scaler for inference
    joblib.dump(scaler, "scaler.pkl")
    # Save feature names to ensure correct order in inference
    joblib.dump(X.columns.tolist(), "model_columns.pkl")

    return X, y, df


# ================== PLOTTING ==================
def plot_distributions(df):
    """Plot churn distributions for tenure and monthly charges."""
    tenure_No = df[df.Churn == 0].tenure
    tenure_Yes = df[df.Churn == 1].tenure
    plt.hist([tenure_No, tenure_Yes], color=['red', 'green'], label=['Churn==No', 'Churn==Yes'])
    plt.legend(); plt.xlabel("Tenure"); plt.ylabel("No of Customers")
    plt.title("Tenure vs Customers")
    plt.show()

    charges_No = df[df.Churn == 0].MonthlyCharges
    charges_Yes = df[df.Churn == 1].MonthlyCharges
    plt.hist([charges_No, charges_Yes], color=['red', 'green'], label=['Churn==No', 'Churn==Yes'])
    plt.legend(); plt.xlabel("MonthlyCharges"); plt.ylabel("No of Customers")
    plt.title("MonthlyCharges vs Customers")
    plt.show()


# ================== MODEL TRAINING ==================
def build_and_train_ann(X_train, y_train, X_test, y_test,
                        loss="binary_crossentropy", batch_size=100,
                        weights=-1, verbose=1):
    """Build, train and evaluate ANN."""
    # Convert inputs
    X_train = np.asarray(X_train).astype(np.float32)
    y_train = np.asarray(y_train).astype(np.float32)
    X_test = np.asarray(X_test).astype(np.float32)
    y_test = np.asarray(y_test).astype(np.float32)

    input_shape = (X_train.shape[1],)

    # Define model
    model = keras.Sequential([
        keras.layers.Dense(23, input_shape=input_shape, activation='relu'),
        keras.layers.Dense(15, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    # Dataset preparation
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

    # Training
    if weights == -1:
        history = model.fit(train_dataset, epochs=2000, callbacks=[early_stop],
                            verbose=verbose, validation_data=test_dataset)
    else:
        history = model.fit(train_dataset, epochs=100, callbacks=[early_stop],
                            verbose=verbose, validation_data=test_dataset)

    # Evaluation
    test_loss, test_acc = model.evaluate(test_dataset, verbose=verbose)
    print(f"\nTest Accuracy: {test_acc:.2f}")

    y_preds = np.round(model.predict(test_dataset, verbose=verbose))
    print("Classification Report:\n", classification_report(y_test, y_preds))

    return model, history, y_preds

# ================== CLASSICAL ML MODELS ==================

def train_logistic_regression(X_train, y_train, X_test, y_test):
    print("\n=== Logistic Regression ===")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_preds = model.predict(X_test)

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_preds))
    print("Classification Report:\n", classification_report(y_test, y_preds))
    print(f"Accuracy: {accuracy_score(y_test, y_preds):.2f}")
    return model, y_preds

def train_random_forest(X_train, y_train, X_test, y_test):
    print("\n=== Random Forest ===")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_preds = model.predict(X_test)

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_preds))
    print("Classification Report:\n", classification_report(y_test, y_preds))
    print(f"Accuracy: {accuracy_score(y_test, y_preds):.2f}")
    return model, y_preds

def train_xgboost(X_train, y_train, X_test, y_test):
    print("\n=== XGBoost ===")
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    y_preds = model.predict(X_test)

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_preds))
    print("Classification Report:\n", classification_report(y_test, y_preds))
    print(f"Accuracy: {accuracy_score(y_test, y_preds):.2f}")
    return model, y_preds



# ================== RESAMPLING METHODS ==================
def undersample(df):
    count_class_0, count_class_1 = df.Churn.value_counts()
    df_class_0 = df[df['Churn'] == 0]
    df_class_1 = df[df['Churn'] == 1]
    df_class_0_under = df_class_0.sample(count_class_1)
    return pd.concat([df_class_0_under, df_class_1], axis=0)


def oversample(df):
    count_class_0, count_class_1 = df.Churn.value_counts()
    df_class_0 = df[df['Churn'] == 0]
    df_class_1 = df[df['Churn'] == 1]
    df_class_1_over = df_class_1.sample(count_class_0, replace=True)
    return pd.concat([df_class_0, df_class_1_over], axis=0)


def smote_sample(X, y):
    smote = SMOTE(sampling_strategy='minority')
    return smote.fit_resample(X, y)


# ================== MODEL COMPARISON ==================
def evaluate_models(X_train, y_train, X_test, y_test):
    results = []

    # Logistic Regression
    log_reg, y_pred_lr = train_logistic_regression(X_train, y_train, X_test, y_test)
    results.append({
        "ModelName": "Logistic Regression",
        "ModelObj": log_reg,
        "Accuracy": accuracy_score(y_test, y_pred_lr),
        "Precision": precision_score(y_test, y_pred_lr),
        "Recall": recall_score(y_test, y_pred_lr),
        "F1": f1_score(y_test, y_pred_lr)
    })

    # Random Forest
    rf, y_pred_rf = train_random_forest(X_train, y_train, X_test, y_test)
    results.append({
        "ModelName": "Random Forest",
        "ModelObj": rf,
        "Accuracy": accuracy_score(y_test, y_pred_rf),
        "Precision": precision_score(y_test, y_pred_rf),
        "Recall": recall_score(y_test, y_pred_rf),
        "F1": f1_score(y_test, y_pred_rf)
    })

    # XGBoost
    xgb, y_pred_xgb = train_xgboost(X_train, y_train, X_test, y_test)
    results.append({
        "ModelName": "XGBoost",
        "ModelObj": xgb,
        "Accuracy": accuracy_score(y_test, y_pred_xgb),
        "Precision": precision_score(y_test, y_pred_xgb),
        "Recall": recall_score(y_test, y_pred_xgb),
        "F1": f1_score(y_test, y_pred_xgb)
    })

    # ANN
    ann, _, y_pred_ann = build_and_train_ann(X_train, y_train, X_test, y_test, verbose=0)
    results.append({
        "ModelName": "ANN",
        "ModelObj": ann,
        "Accuracy": accuracy_score(y_test, y_pred_ann),
        "Precision": precision_score(y_test, y_pred_ann),
        "Recall": recall_score(y_test, y_pred_ann),
        "F1": f1_score(y_test, y_pred_ann)
    })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    print("\n=== Model Comparison ===")
    print(results_df)

    # Pick best model by Accuracy
    best_row = results_df.loc[results_df['Accuracy'].idxmax()]
    print(f"\nBest Model: {best_row['ModelName']}")
    print(f"Accuracy: {best_row['Accuracy']:.2f}")
    print(f"Precision: {best_row['Precision']:.2f}")
    
    # Assuming 'best_model' is your trained model
    joblib.dump(best_row['ModelObj'], "best_model.pkl")

# ================== MAIN ==================
def main():
    X, y, df = preprocess_data("churn.csv")

    # Plot initial distributions
    plot_distributions(df)

    # Baseline training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42, stratify=y)
    print("\n=== Baseline Model ===")
    build_and_train_ann(X_train, y_train, X_test, y_test, batch_size=1500, verbose=1)

    # Classical models for baseline comparison
    print("\n=== Classical ML Models ===")
    train_logistic_regression(X_train, y_train, X_test, y_test)
    train_random_forest(X_train, y_train, X_test, y_test)
    train_xgboost(X_train, y_train, X_test, y_test)

    # Undersampling
    print("\n=== Undersampling ===")
    df_under = undersample(df)
    X, y = df_under.drop('Churn', axis=1), df_under['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=15, stratify=y)
    build_and_train_ann(X_train, y_train, X_test, y_test, verbose=1)

    # Oversampling
    print("\n=== Oversampling ===")
    df_over = oversample(df)
    X, y = df_over.drop('Churn', axis=1), df_over['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=15, stratify=y)
    build_and_train_ann(X_train, y_train, X_test, y_test, verbose=1)

    # SMOTE
    print("\n=== SMOTE Sampling ===")
    X_sm, y_sm = smote_sample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.2,
                                                        random_state=15, stratify=y_sm)
    build_and_train_ann(X_train, y_train, X_test, y_test, verbose=1)

     # Compare models
    evaluate_models(X_train, y_train, X_test, y_test)

    


if __name__ == "__main__":
    main()
