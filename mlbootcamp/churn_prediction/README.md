# Customer Churn Prediction - MLE Mini Project

## Overview
This project demonstrates a complete machine learning pipeline for predicting customer churn in a telecommunications service. The notebook covers the entire ML workflow from data exploration to model deployment considerations.

## Project Structure
- `Student_MLE_MiniProject_Churn_Prediction.ipynb` - Main Jupyter notebook with complete analysis
- `churn_aws_pipeline.ipynb` - Local baseline pipeline exploration (optional)
- `preprocess.py` - Feature engineering script used by SageMaker Processing step
- `run_sagemaker_pipeline.py` - Defines and runs a minimal SageMaker Pipeline (Processing → Train → Register + Batch Transform)
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Features
- **Data Generation**: Creates realistic synthetic telecom churn data
- **Exploratory Data Analysis**: Comprehensive visualization and statistical analysis
- **Feature Engineering**: Creates new features and handles categorical variables
- **Model Comparison**: Tests multiple ML algorithms (Logistic Regression, Random Forest, Gradient Boosting, SVM)
- **Hyperparameter Tuning**: Uses GridSearchCV to optimize the best model
- **Business Insights**: Provides actionable recommendations for customer retention

## Key Sections
1. **Data Loading & Exploration** - Understanding the dataset structure
2. **Target Variable Analysis** - Examining churn distribution and class imbalance
3. **Feature Analysis** - Visualizing relationships between features and churn
4. **Data Preprocessing** - Handling missing values and encoding variables
5. **Model Training** - Training and evaluating multiple models
6. **Feature Importance** - Understanding key factors influencing churn
7. **Model Tuning** - Optimizing hyperparameters
8. **Final Evaluation** - Confusion matrix, ROC curves, and classification reports
9. **Business Insights** - Actionable recommendations for retention strategies

## How to Run (local)
1. Install dependencies: `pip install -r requirements.txt`
2. Open the notebook: `jupyter notebook Student_MLE_MiniProject_Churn_Prediction.ipynb`
3. Run all cells sequentially

## How to Run (AWS SageMaker Pipelines)
Prereqs: AWS account, SageMaker execution role, S3 bucket, correct region.

1. Export AWS config (example):
   - `export AWS_REGION=us-west-2`
   - `export SAGEMAKER_ROLE_ARN=arn:aws:iam::<ACCOUNT_ID>:role/service-role/AmazonSageMaker-ExecutionRole-<DATE>`
2. Create/update pipeline:
   - `python run_sagemaker_pipeline.py --bucket <your-s3-bucket> --action create`
3. Start a run:
   - `python run_sagemaker_pipeline.py --bucket <your-s3-bucket> --action run`
4. Monitor in SageMaker Studio: Pipeline graph, steps, and artifacts.

This aligns with the Springboard mini-project brief and the AWS blog pattern for an end-to-end churn pipeline (Processing, Training/XGBoost, quality gate, registration, and batch transform).

## Expected Outcomes
- Understanding of customer churn patterns
- Comparison of different ML model performances
- Identification of key churn factors
- Business recommendations for customer retention
- Complete ML pipeline demonstration

## Student Information
- **Name**: Luke Johnson
- **Date**: January 2025
- **Course**: MLE Mini Project
