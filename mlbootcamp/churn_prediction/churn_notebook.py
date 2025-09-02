# Mini Project: Customer Churn Prediction
# Student: Luke Johnson
# Date: January 2025
# Course: MLE Mini Project

# Cell 1: Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
try:
    plt.style.use('seaborn-v0_8')
except Exception:
    plt.style.use('seaborn-darkgrid')
sns.set_palette("husl")
np.random.seed(42)

# Cell 2: Data Generation
# Generate synthetic telecom churn data
n_samples = 7043

# Customer demographics
customer_ids = [f'CUST_{i:06d}' for i in range(1, n_samples + 1)]
genders = np.random.choice(['Male', 'Female'], n_samples)
senior_citizens = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
partners = np.random.choice(['Yes', 'No'], n_samples, p=[0.5, 0.5])
dependents = np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7])

# Service features
tenures = np.random.randint(1, 73, n_samples)  # 1-72 months
phone_services = np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1])
internet_services = np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.4, 0.4, 0.2])

# Additional services
online_security = np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.4, 0.3])
online_backup = np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.4, 0.3])
device_protection = np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.4, 0.3])
tech_support = np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.4, 0.3])
streaming_tv = np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.4, 0.3, 0.3])
streaming_movies = np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.4, 0.3, 0.3])

# Contract and billing
contracts = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.6, 0.25, 0.15])
paperless_billing = np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4])
payment_methods = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], 
                                   n_samples, p=[0.3, 0.2, 0.25, 0.25])

# Financial features
monthly_charges = np.random.uniform(20, 120, n_samples)
total_charges = monthly_charges * tenures + np.random.uniform(-50, 100, n_samples)

# Create churn target (with realistic patterns)
churn_probs = np.zeros(n_samples)
for i in range(n_samples):
    base_prob = 0.2
    
    # Higher churn for month-to-month contracts
    if contracts[i] == 'Month-to-month':
        base_prob += 0.15
    elif contracts[i] == 'One year':
        base_prob += 0.05
    
    # Higher churn for higher monthly charges
    if monthly_charges[i] > 80:
        base_prob += 0.1
    
    # Higher churn for shorter tenures
    if tenures[i] < 12:
        base_prob += 0.1
    
    # Higher churn for electronic check payments
    if payment_methods[i] == 'Electronic check':
        base_prob += 0.05
    
    churn_probs[i] = min(base_prob, 0.6)

churns = np.random.binomial(1, churn_probs, n_samples)
churn_labels = ['No' if x == 0 else 'Yes' for x in churns]

# Create DataFrame
data = pd.DataFrame({
    'customerID': customer_ids,
    'gender': genders,
    'SeniorCitizen': senior_citizens,
    'Partner': partners,
    'Dependents': dependents,
    'tenure': tenures,
    'PhoneService': phone_services,
    'InternetService': internet_services,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contracts,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_methods,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'Churn': churn_labels
})

print(f"Dataset shape: {data.shape}")
print(f"Columns: {list(data.columns)}")

# Cell 3: Data Exploration
print("\nFirst few rows:")
print(data.head())

print("\nData types:")
print(data.dtypes)

print("\nMissing values:")
print(data.isnull().sum())

print("\nBasic statistics:")
print(data.describe())

# Cell 4: Target Variable Analysis
print("\nChurn distribution:")
churn_counts = data['Churn'].value_counts()
print(churn_counts)
print(f"\nChurn rate: {churn_counts['Yes'] / len(data) * 100:.2f}%")

# Visualize churn distribution
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
churn_counts.plot(kind='bar', color=['lightblue', 'lightcoral'])
plt.title('Churn Distribution')
plt.xlabel('Churn')
plt.ylabel('Count')
plt.xticks(rotation=0)

plt.subplot(1, 2, 2)
plt.pie(churn_counts.values, labels=churn_counts.index, autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
plt.title('Churn Rate')
plt.show()

# Cell 5: Feature Analysis
# Numerical features
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
                       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                       'PaperlessBilling', 'PaymentMethod']

print("\nNumerical features statistics:")
print(data[numerical_features].describe())

# Visualize numerical features by churn
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, feature in enumerate(numerical_features):
    axes[i].hist(data[data['Churn'] == 'No'][feature], alpha=0.7, label='No Churn', bins=30)
    axes[i].hist(data[data['Churn'] == 'Yes'][feature], alpha=0.7, label='Churn', bins=30)
    axes[i].set_title(f'{feature} Distribution by Churn')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Frequency')
    axes[i].legend()
plt.tight_layout()
plt.show()

# Cell 6: Categorical Feature Analysis
# Analyze key categorical features
key_categorical = ['Contract', 'InternetService', 'PaymentMethod', 'tenure_group']

# Create tenure groups
data['tenure_group'] = pd.cut(data['tenure'], bins=[0, 12, 24, 48, 72], 
                              labels=['0-12 months', '13-24 months', '25-48 months', '49+ months'])

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

for i, feature in enumerate(key_categorical):
    if feature in data.columns:
        churn_by_feature = data.groupby([feature, 'Churn']).size().unstack(fill_value=0)
        churn_by_feature.plot(kind='bar', ax=axes[i], stacked=True)
        axes[i].set_title(f'Churn by {feature}')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Count')
        axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Cell 7: Correlation Analysis
# Convert categorical variables to numerical for correlation
data_encoded = data.copy()
le = LabelEncoder()

for col in categorical_features + ['tenure_group']:
    if col in data_encoded.columns:
        data_encoded[col] = le.fit_transform(data_encoded[col].astype(str))

# Convert target to binary
data_encoded['Churn_binary'] = (data_encoded['Churn'] == 'Yes').astype(int)

# Calculate correlations
correlation_matrix = data_encoded.drop(['customerID', 'Churn'], axis=1).corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# Cell 8: Data Preprocessing
# Remove customerID (not useful for prediction)
data_processed = data.drop('customerID', axis=1)

# Handle missing values
print("\nMissing values before processing:")
print(data_processed.isnull().sum())

# Fill missing values
data_processed['TotalCharges'] = data_processed['TotalCharges'].fillna(data_processed['TotalCharges'].median())

print("\nMissing values after processing:")
print(data_processed.isnull().sum())

# Cell 9: Feature Engineering
# Create new features
data_processed['tenure_months'] = data_processed['tenure']
data_processed['monthly_to_total_ratio'] = data_processed['MonthlyCharges'] / (data_processed['TotalCharges'] + 1)

# Create service count features
service_columns = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
data_processed['total_services'] = data_processed[service_columns].apply(lambda x: (x == 'Yes').sum(), axis=1)

# Create contract type encoding
contract_mapping = {'Month-to-month': 1, 'One year': 2, 'Two year': 3}
data_processed['contract_numeric'] = data_processed['Contract'].map(contract_mapping)

print("\nNew features created:")
print(data_processed[['tenure_months', 'monthly_to_total_ratio', 'total_services', 'contract_numeric']].head())

# Cell 10: Prepare Data for Modeling
# Separate features and target
X = data_processed.drop('Churn', axis=1)
y = data_processed['Churn']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")
print(f"Training churn rate: {y_train.value_counts(normalize=True)['Yes']:.2f}")
print(f"Test churn rate: {y_test.value_counts(normalize=True)['Yes']:.2f}")

# Cell 11: Handle Categorical Variables
# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

print(f"\nCategorical columns: {list(categorical_cols)}")
print(f"Numerical columns: {list(numerical_cols)}")

# Create encoders
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', sparse=False), categorical_cols)
    ])

# Fit and transform training data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"\nProcessed training set shape: {X_train_processed.shape}")
print(f"Processed test set shape: {X_test_processed.shape}")

# Cell 12: Handle Class Imbalance
print("\nClass distribution before balancing:")
print(pd.Series(y_train).value_counts())

# Apply SMOTE to balance classes
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_processed, y_train)

print("\nClass distribution after SMOTE:")
print(pd.Series(y_train_balanced).value_counts())

# Cell 13: Baseline Models
# Define models to test
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(random_state=42, probability=True)
}

# Train and evaluate baseline models
baseline_results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train model
    model.fit(X_train_balanced, y_train_balanced)
    
    # Make predictions
    y_pred = model.predict(X_test_processed)
    y_pred_proba = model.predict_proba(X_test_processed)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='Yes')
    recall = recall_score(y_test, y_pred, pos_label='Yes')
    f1 = f1_score(y_test, y_pred, pos_label='Yes')
    auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    
    baseline_results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }
    
    print(f"{name} Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  AUC: {auc:.4f}" if auc else "  AUC: N/A")

# Cell 14: Model Comparison
# Create comparison DataFrame
results_df = pd.DataFrame(baseline_results).T
print("\nModel Comparison:")
print(results_df.round(4))

# Visualize results
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

metrics = ['accuracy', 'precision', 'recall', 'f1']
for i, metric in enumerate(metrics):
    axes[i].bar(results_df.index, results_df[metric])
    axes[i].set_title(f'{metric.capitalize()} Comparison')
    axes[i].set_ylabel(metric.capitalize())
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Cell 15: Feature Importance (Random Forest)
# Get feature names after preprocessing (robust to sklearn versions)
try:
    feature_names = list(preprocessor.get_feature_names_out())
except Exception:
    num_names = list(numerical_cols)
    ohe = preprocessor.named_transformers_['cat']
    if hasattr(ohe, 'get_feature_names_out'):
        cat_names = list(ohe.get_feature_names_out(list(categorical_cols)))
    else:
        cat_names = [f"{col}_{cat}" for col, cats in zip(categorical_cols, ohe.categories_) for cat in cats[1:]]
    feature_names = num_names + cat_names

# Get feature importance from Random Forest
rf_model = models['Random Forest']
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 Most Important Features:")
print(feature_importance.head(15))

# Visualize feature importance
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance')
plt.title('Top 15 Feature Importances (Random Forest)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Cell 16: Hyperparameter Tuning
# Tune the best performing model (Random Forest)
print("\nHyperparameter tuning for Random Forest...")

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_tuned = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf_tuned, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
grid_search.fit(X_train_balanced, y_train_balanced)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Cell 17: Final Model Evaluation
# Get best model
best_rf = grid_search.best_estimator_

# Make predictions with tuned model
y_pred_tuned = best_rf.predict(X_test_processed)
y_pred_proba_tuned = best_rf.predict_proba(X_test_processed)[:, 1]

# Calculate metrics
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
precision_tuned = precision_score(y_test, y_pred_tuned, pos_label='Yes')
recall_tuned = recall_score(y_test, y_pred_tuned, pos_label='Yes')
f1_tuned = f1_score(y_test, y_pred_tuned, pos_label='Yes')
auc_tuned = roc_auc_score(y_test, y_pred_proba_tuned)

print("\nTuned Random Forest Results:")
print(f"Accuracy: {accuracy_tuned:.4f}")
print(f"Precision: {precision_tuned:.4f}")
print(f"Recall: {recall_tuned:.4f}")
print(f"F1-Score: {f1_tuned:.4f}")
print(f"AUC: {auc_tuned:.4f}")

# Cell 18: Confusion Matrix and Classification Report
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_tuned)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Churn', 'Churn'], 
            yticklabels=['No Churn', 'Churn'])
plt.title('Confusion Matrix - Tuned Random Forest')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_tuned, target_names=['No Churn', 'Churn']))

# Cell 19: ROC Curve
# Calculate ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_tuned)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_tuned:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Cell 20: Business Insights and Recommendations
print("\n=== BUSINESS INSIGHTS AND RECOMMENDATIONS ===")
print("\nKey Findings:")

# Analyze top features
top_5_features = feature_importance.head(5)
print("\nTop 5 factors influencing churn:")
for i, (_, row) in enumerate(top_5_features.iterrows(), 1):
    print(f"{i}. {row['feature']}: {row['importance']:.4f}")

# Contract analysis
contract_churn = data.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean()).sort_values(ascending=False)
print("\nChurn rate by contract type:")
for contract, rate in contract_churn.items():
    print(f"  {contract}: {rate:.3f}")

# Tenure analysis
tenure_churn = data.groupby('tenure_group')['Churn'].apply(lambda x: (x == 'Yes').mean())
print("\nChurn rate by tenure:")
for tenure, rate in tenure_churn.items():
    print(f"  {tenure}: {rate:.3f}")

print("\n\nRecommendations:")
print("1. Focus retention efforts on month-to-month contract customers")
print("2. Implement loyalty programs for customers with tenure < 12 months")
print("3. Review pricing strategy for high monthly charge customers")
print("4. Improve payment method options to reduce electronic check usage")
print("5. Enhance service quality for customers with multiple add-on services")

# Cell 21: Model Deployment Considerations
print("\n=== MODEL DEPLOYMENT CONSIDERATIONS ===")
print("\nModel Performance Summary:")
print(f"Final Model: Tuned Random Forest")
print(f"Accuracy: {accuracy_tuned:.4f}")
print(f"F1-Score: {f1_tuned:.4f}")
print(f"AUC: {auc_tuned:.4f}")

print("\nDeployment Recommendations:")
print("1. Monitor model performance monthly")
print("2. Retrain model quarterly with new data")
print("3. Implement A/B testing for different retention strategies")
print("4. Set up automated alerts for high-churn-risk customers")
print("5. Create dashboard for business stakeholders")

print("\nSuccess Metrics:")
print("- Reduction in customer churn rate")
print("- Increase in customer lifetime value")
print("- Improvement in retention campaign effectiveness")
print("- Cost savings from proactive retention efforts")

print("\n=== PROJECT COMPLETED ===")
print("This notebook demonstrates a complete machine learning pipeline for customer churn prediction.")
print("The tuned Random Forest model achieved good performance and provides actionable business insights.")
