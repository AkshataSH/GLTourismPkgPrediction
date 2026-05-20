# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
from pathlib import Path
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

HF_REPO_ID = "akshatash/GLTourismPkgPrediction"
DATASET_PATH = "hf://datasets/akshatash/GLTourismPkgPrediction/tourism.csv"
OUTPUT_DIR = Path("data/processed")
TARGET_COL = "ProdTaken"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

token = os.getenv("HF_TOKEN")
if not token:
    raise ValueError("HF_TOKEN environment variable is not set.")

api = HfApi(token=token)

df = pd.read_csv(DATASET_PATH)
print("Dataset loaded:", df.shape)

# Display the first few rows of the dataset and last few rows of the dataset to understand its structure
print("First 5 rows of the dataset:")
print(df.head())
print ("-"*50)
print("First 5 rows of the dataset:")
print(df.tail())
# Display the shape of the dataset
print("\nShape of the dataset:", df.shape) #The dataset contains 4128 rows with 21 columns
# Display the data types of each column
print("\nData types of each column:")
print(df.dtypes)
print("\nDescriptive statistics of the dataset:")
print(df.describe(include='all').T)
# Display the number of missing values in each column
print("\nNumber of missing values in each column:")
print(df.isnull().sum())
# checking for unique values in ID column
print("\nNumber of unique values in each column:")
print(df.nunique()) #Unnamed and Customer ID have unique values. Unnamed is an index column, no analytical value. CustomerID is an unique identifier for each customer, doesn’t help prediction.
# Drop unique identifier column (not useful for modeling)
df.drop(columns=['Unnamed: 0', 'CustomerID'], inplace=True)
print("\nShape of the dataset after dropping unique identifier columns:", df.shape)

# Handle specific data quality issues (e.g., "Fe Male" should be "Female")
df["Gender"] = df["Gender"].astype(str).str.strip().replace({"Fe Male": "Female", "Fe male": "Female"})

# Data Cleaning and Feature Engineering
# Handle missing values - for simplicity, fill with appropriate values
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['MonthlyIncome'].fillna(df['MonthlyIncome'].mean(), inplace=True)

# Convert categorical features to numerical
df['TypeofContact'] = df['TypeofContact'].map({'Company Invited': 0, 'Self Enquiry': 1, 'Self Inquiry': 1})
df['CityTier'] = df['CityTier'] - 1  # Convert to 0, 1, 2
df['Occupation'] = df['Occupation'].map({
    'Salaried': 0,
    'Free Lancer': 1,
    'Freelancer': 1,
    'Small Business': 2,
    'Large Business': 3
})
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['MaritalStatus'] = df['MaritalStatus'].map({'Single': 0, 'Married': 1, 'Divorced': 2, 'Unmarried': 3})

# Split the dataset into training and testing sets
X = df.drop('ProdTaken', axis=1)
y = df['ProdTaken']
print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target distribution:\n{y.value_counts()}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
print(f"\nTrain set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

X_train.to_csv("Xtrain.csv", index=False)
X_test.to_csv("Xtest.csv", index=False)
y_train.to_csv("ytrain.csv", index=False)
y_test.to_csv("ytest.csv", index=False)

files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="akshatash/GLTourismPkgPrediction",
        repo_type="dataset",
    )
