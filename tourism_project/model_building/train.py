# Data manipulation
import pandas as pd
import numpy as np

# Preprocessing & pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# Model training & evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)

# Serialization & system
import joblib
import os

# Hugging Face Hub
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# MLflow
import mlflow
# Set up MLflow tracking
mlflow.set_tracking_uri("https://undefined-illusive-dole.ngrok-free.dev")
mlflow.set_experiment("tourism-package-prediction")
repo_id = "akshatash/GLTourismPkgPrediction"
api = HfApi()

# Load the preprocessed data from Hugging Face
Xtrain_path = f"hf://datasets/{repo_id}/Xtrain.csv"
Xtest_path = f"hf://datasets/{repo_id}/Xtest.csv"
ytrain_path = f"hf://datasets/{repo_id}/ytrain.csv"
ytest_path = f"hf://datasets/{repo_id}/ytest.csv"

print("Loading preprocessed data...")
X_train = pd.read_csv(Xtrain_path)
X_test = pd.read_csv(Xtest_path)
y_train = pd.read_csv(ytrain_path).values.ravel()
y_test = pd.read_csv(ytest_path).values.ravel()

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Automatically detect columns
numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()
scale_pos_weight = round(negative_count / positive_count, 2)


# Define XGBoost Regressor
xgb_model = xgb.XGBClassifier(random_state=42,
    eval_metric='logloss')

# Define hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [100, 200,300],
    'xgbclassifier__max_depth': [2, 3,5],
    'xgbclassifier__learning_rate': [0.03, 0.05, 0.1],
    'xgbclassifier__subsample': [0.5],
    'xgbclassifier__colsample_bytree': [0.5], 
    'xgbclassifier__reg_lambda': [0.5]  
}

# Create pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

mlflow.set_experiment("tourism-package-prediction")
with mlflow.start_run():
    print("Performing Grid Search with Cross-Validation...")
    grid_search = GridSearchCV(
        model_pipeline, 
        param_grid, 
        cv=5, 
        n_jobs=-1, 
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    # Log parameter sets
    results = grid_search.cv_results_
    print(f"\nEvaluated {len(results['params'])} parameter combinations")

    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_roc_auc", mean_score)

    # Best model
    print(f"\nBest parameters: {grid_search.best_params_}")
    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Predictions
    print("\nMaking predictions...")
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    # Probability predictions
    y_pred_train_proba = best_model.predict_proba(X_train)[:, 1]
    y_pred_test_proba = best_model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    print("\nCalculating metrics...")
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    train_precision = precision_score(y_train, y_pred_train, zero_division=0)
    test_precision = precision_score(y_test, y_pred_test, zero_division=0)

    train_recall = recall_score(y_train, y_pred_train, zero_division=0)
    test_recall = recall_score(y_test, y_pred_test, zero_division=0)

    train_f1 = f1_score(y_train, y_pred_train, zero_division=0)
    test_f1 = f1_score(y_test, y_pred_test, zero_division=0)

    train_roc_auc = roc_auc_score(y_train, y_pred_train_proba)
    test_roc_auc = roc_auc_score(y_test, y_pred_test_proba)

    # Log metrics
    mlflow.log_metrics({
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "train_precision": train_precision,
        "test_precision": test_precision,
        "train_recall": train_recall,
        "test_recall": test_recall,
        "train_f1_score": train_f1,
        "test_f1_score": test_f1,
        "train_roc_auc": train_roc_auc,
        "test_roc_auc": test_roc_auc
    })

    # Print results
    print("\n" + "="*50)
    print("MODEL PERFORMANCE METRICS")
    print("="*50)
    print(f"Train Accuracy: {train_accuracy:.4f} | Test Accuracy: {test_accuracy:.4f}")
    print(f"Train Precision: {train_precision:.4f} | Test Precision: {test_precision:.4f}")
    print(f"Train Recall: {train_recall:.4f} | Test Recall: {test_recall:.4f}")
    print(f"Train F1-Score: {train_f1:.4f} | Test F1-Score: {test_f1:.4f}")
    print(f"Train ROC-AUC: {train_roc_auc:.4f} | Test ROC-AUC: {test_roc_auc:.4f}")
    print("="*50)

    print("\nTest Set Classification Report:")
    print(classification_report(y_test, y_pred_test, target_names=['No Purchase', 'Purchase']))

    print("\nTest Set Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_test))

    model_path = "best_tourism_model_v1.joblib"
    joblib.dump(best_model, model_path)
    print(f"\nModel saved locally as: {model_path}")

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model logged to MLflow")

    # Upload to Hugging Face
    repo_type = "model"

    # Step 1: Check if the repository exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"\nRepository '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"\nRepository '{repo_id}' not found. Creating new repository...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Repository '{repo_id}' created.")

    # Upload model to Hugging Face
    api.upload_file(
        path_or_fileobj="best_tourism_model_v1.joblib",
        path_in_repo="best_tourism_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print(f"Model uploaded to Hugging Face: {repo_id}")

print("\n" + "="*50)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("="*50)
