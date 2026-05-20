# MlOps1Final

This repository contains an Azure Machine Learning and MLOps project focused on a tourism dataset and associated model building, deployment, and hosting workflows.

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
