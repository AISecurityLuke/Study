# Student MLE MiniProject: Fine-Tuning

## Overview
This project demonstrates fine-tuning a pre-trained transformer model (DistilBERT) for binary text classification. The notebook covers the complete machine learning pipeline from data preparation to model deployment.

## Project Structure
```
fine_tuning/
├── Student_MLE_MiniProject_Fine_Tuning.ipynb  # Main notebook
├── requirements.txt                            # Dependencies
├── README.md                                  # This file
└── .keep                                      # Directory placeholder
```

## Features
- **Complete ML Pipeline**: Data loading, preprocessing, model training, evaluation, and deployment
- **Transformer Fine-tuning**: Uses Hugging Face Transformers library with DistilBERT
- **Comprehensive Evaluation**: Accuracy, precision, recall, F1-score, confusion matrix
- **Model Analysis**: Confidence analysis, error analysis, and insights
- **Production Ready**: Model saving/loading and inference testing

## Requirements
Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Open the Jupyter notebook: `Student_MLE_MiniProject_Fine_Tuning.ipynb`
2. Run cells sequentially from top to bottom
3. The notebook will:
   - Create a synthetic dataset for demonstration
   - Fine-tune a DistilBERT model
   - Evaluate performance
   - Save the trained model
   - Test model loading and inference

## Key Sections
1. **Setup and Imports**: Install and import required libraries
2. **Data Loading**: Create synthetic sentiment analysis dataset
3. **Data Preprocessing**: Split data and create PyTorch datasets
4. **Model Setup**: Load pre-trained DistilBERT model
5. **Training**: Configure and run fine-tuning
6. **Evaluation**: Assess model performance on test set
7. **Analysis**: Confidence analysis and error investigation
8. **Deployment**: Save model and test loading

## Model Details
- **Base Model**: `distilbert-base-uncased`
- **Task**: Binary text classification (sentiment analysis)
- **Architecture**: DistilBERT with classification head
- **Training**: 3 epochs, learning rate 2e-5, batch size 16

## Outputs
- Trained model saved to `./fine_tuned_model/`
- Training results saved to `./training_results.json`
- Logs saved to `./logs/`
- Results saved to `./results/`

## Notes
- This notebook uses synthetic data for demonstration
- In production, replace with your actual dataset
- Adjust hyperparameters based on your specific use case
- The model can be easily adapted for other classification tasks

## Author
Luke Johnson - Machine Learning Engineering Student
