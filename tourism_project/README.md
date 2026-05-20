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

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:
   ```bash
   python -m pip install -r tourism_project/requirements.txt
   ```

## Usage

- Use `Learner_Template_Notebook_AML_and_MLOps_Project.ipynb` for step-by-step exploration.
- Run training and data preparation scripts from `tourism_project/model_building/`.
- Deploy the app using the `tourism_project/deployment/Dockerfile` and `tourism_project/deployment/app.py`.
- Use `tourism_project/hosting/hosting.py` for model hosting workflows.

## Notes

- This repository is intended for learning and experimenting with MLOps pipeline components.
- Update the README as project scope and scripts evolve.
