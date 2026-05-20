# Advanced Machine Learning and MLOps: Tourism Package Prediction

## Business Context
"Visit with Us," a leading travel company, is revolutionizing the tourism industry by leveraging data-driven strategies to optimize operations and customer engagement. While introducing a new package offering, such as the Wellness Tourism Package, the company faces challenges in targeting the right customers efficiently. The manual approach to identifying potential customers is inconsistent, time-consuming, and prone to errors, leading to missed opportunities and suboptimal campaign performance.

To address these issues, the company aims to implement a scalable and automated system that integrates customer data, predicts potential buyers, and enhances decision-making for marketing strategies. By utilizing an MLOps pipeline, the company seeks to achieve seamless integration of data preprocessing, model development, deployment, and CI/CD practices for continuous improvement. This system will ensure efficient targeting of customers, timely updates to the predictive model, and adaptation to evolving customer behaviors, ultimately driving growth and customer satisfaction.

## Objective
As an MLOps Engineer at "Visit with Us," your responsibility is to design and deploy an MLOps pipeline on GitHub to automate the end-to-end workflow for predicting customer purchases. The primary objective is to build a model that predicts whether a customer will purchase the newly introduced Wellness Tourism Package before contacting them. The pipeline will include data cleaning, preprocessing, transformation, model building, training, evaluation, and deployment, ensuring consistent performance and scalability. By leveraging GitHub Actions for CI/CD integration, the system will enable automated updates, streamline model deployment, and improve operational efficiency. This robust predictive solution will empower policymakers to make data-driven decisions, enhance marketing strategies, and effectively target potential customers, thereby driving customer acquisition and business growth.

This repository contains an Machine Learning and MLOps project focused on a tourism dataset and associated model building, deployment, and hosting workflows.

## Project Structure

- `Learner_Template_Notebook_AML_and_MLOps_Project.ipynb` - Jupyter notebook for learning and exploring Azure ML and MLOps concepts.
- `tourism_project/` - Main project folder containing code, data, deployment, and hosting resources.
  - `data/tourism.csv` - Dataset used for training and experimentation.
  - `model_building/` - Scripts for preparing data, training models, and registering artifacts.
    - `prep.py` - Data preparation logic.
    - `train.py` - Model training script.
    - `data_register.py` - Data registration script.
  - `deployment/` - Application deployment resources.
    - `app.py` - Flask application for serving the trained model.
    - `Dockerfile` - Container image definition.
    - `requirements.txt` - Python dependencies required for deployment.
  - `hosting/` - Hosting and inference scripts.
    - `hosting.py` - Hosting logic for deploying and serving the model.
  - `requirements.txt` - Project-level dependencies for local development and model building.

## Problem Statement

The goal is to classify customers into:

- `0`: No Purchase
- `1`: Purchase

The target column is `ProdTaken`. The model uses customer demographic, travel, interaction, and package-related features such as age, type of contact, city tier, pitch duration, occupation, product pitched, passport status, car ownership, designation, and monthly income.

## Dataset

The raw dataset is stored at:

```text
data/tourism.csv
```

The project also uses Hugging Face repositories for data and model artifacts:

- Dataset/model repository: `akshatash/GLTourismPkgPrediction`
- Raw dataset path used in preprocessing: `hf://datasets/akshatash/GLTourismPkgPrediction/tourism.csv`
- Trained model file: `best_tourism_model_v1.joblib`

## Workflow

### 1. Register Dataset

`model_building/data_register.py` creates or reuses the Hugging Face dataset repository and uploads the local `data` folder.

```bash
python model_building/data_register.py
```

This step requires a Hugging Face token.

### 2. Preprocess Data

`model_building/prep.py` loads the dataset from Hugging Face, cleans the data, applies basic encoding, splits the data into train and test sets, and uploads the processed files back to the Hugging Face dataset repository.

Generated files:

- `Xtrain.csv`
- `Xtest.csv`
- `ytrain.csv`
- `ytest.csv`

Run:

```bash
python model_building/prep.py
```

Preprocessing includes:

- Dropping identifier columns: `Unnamed: 0`, `CustomerID`
- Fixing gender value inconsistencies such as `Fe Male`
- Filling missing `Age` and `MonthlyIncome` values with their means
- Encoding selected categorical columns
- Splitting data into train and test sets using stratified sampling

### 3. Train Model

`model_building/train.py` loads the processed train/test files from Hugging Face, builds a preprocessing and XGBoost pipeline, performs grid search cross-validation, logs metrics to MLflow, saves the best model, and uploads it to Hugging Face as a model artifact.

```bash
python model_building/train.py
```

The training pipeline includes:

- `StandardScaler` for numeric features
- `OneHotEncoder` for categorical features
- `XGBClassifier`
- `GridSearchCV` for hyperparameter tuning
- MLflow experiment tracking under `tourism-package-prediction`

Tracked metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

### 4. Host Deployment Files

`hosting/hosting.py` uploads the contents of the `deployment` folder to the Hugging Face Space repository.

```bash
python hosting/hosting.py
```

## Streamlit App

The Streamlit app is located at:

```text
deployment/app.py
```

It downloads the trained model and reference dataset from Hugging Face, accepts customer details through a form, prepares the input using the same feature transformations, and returns:

- Purchase prediction
- Prediction label
- Purchase probability, when available from the model
- Prepared model input for inspection

Run locally from the `deployment` folder:

```bash
cd deployment
streamlit run app.py
```

The app is configured for deployment with Docker and runs on port `8501`.

## Docker Deployment

The deployment folder contains a Dockerfile for running the Streamlit app.

Build the image:

```bash
cd deployment
docker build -t tourism-package-predictor .
```

Run the container:

```bash
docker run -p 8501:8501 tourism-package-predictor
```

Then open:

```text
http://localhost:8501
```

## Environment Variables

Set the following environment variable before running scripts that upload files to Hugging Face:

```bash
HF_TOKEN=<your_hugging_face_token>
```

On Windows PowerShell:

```powershell
$env:HF_TOKEN="your_hugging_face_token"
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For deployment-only usage, install dependencies from:

```bash
pip install -r deployment/requirements.txt
```

## Main Dependencies

- pandas
- scikit-learn
- xgboost
- mlflow
- huggingface_hub
- streamlit
- joblib
