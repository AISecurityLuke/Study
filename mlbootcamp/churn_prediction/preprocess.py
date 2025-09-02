import argparse
import os
import numpy as np
import pandas as pd


def generate_synthetic_churn(n_samples: int = 7043, random_seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)

    customer_ids = [f"CUST_{i:06d}" for i in range(1, n_samples + 1)]
    genders = rng.choice(['Male', 'Female'], n_samples)
    senior_citizens = rng.choice([0, 1], n_samples, p=[0.8, 0.2])
    partners = rng.choice(['Yes', 'No'], n_samples, p=[0.5, 0.5])
    dependents = rng.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7])

    tenures = rng.integers(1, 73, n_samples)
    phone_services = rng.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1])
    internet_services = rng.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.4, 0.4, 0.2])

    online_security = rng.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.4, 0.3])
    online_backup = rng.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.4, 0.3])
    device_protection = rng.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.4, 0.3])
    tech_support = rng.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.4, 0.3])
    streaming_tv = rng.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.4, 0.3, 0.3])
    streaming_movies = rng.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.4, 0.3, 0.3])

    contracts = rng.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.6, 0.25, 0.15])
    paperless_billing = rng.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4])
    payment_methods = rng.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n_samples, p=[0.3, 0.2, 0.25, 0.25])

    monthly_charges = rng.uniform(20, 120, n_samples)
    total_charges = monthly_charges * tenures + rng.uniform(-50, 100, n_samples)

    churn_probs = np.zeros(n_samples)
    for i in range(n_samples):
        base_prob = 0.2
        if contracts[i] == 'Month-to-month':
            base_prob += 0.15
        elif contracts[i] == 'One year':
            base_prob += 0.05
        if monthly_charges[i] > 80:
            base_prob += 0.1
        if tenures[i] < 12:
            base_prob += 0.1
        if payment_methods[i] == 'Electronic check':
            base_prob += 0.05
        churn_probs[i] = min(base_prob, 0.6)

    churns = rng.binomial(1, churn_probs, n_samples)

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
        'Churn': churns,
    })
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-output', type=str, default='/opt/ml/processing/train/train.csv')
    parser.add_argument('--test-output', type=str, default='/opt/ml/processing/test/test.csv')
    parser.add_argument('--num-samples', type=int, default=7043)
    args = parser.parse_args()

    df = generate_synthetic_churn(n_samples=args.num_samples, random_seed=42)

    # Basic preprocessing for XGBoost: one-hot encode categoricals, keep numeric, label as target
    y = df['Churn'].astype(int)
    X = df.drop(['customerID', 'Churn'], axis=1).copy()

    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    X_enc['target'] = y.values

    # Train/Test split (simple holdout)
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(X_enc, test_size=0.2, random_state=42, stratify=y)

    # Ensure output dirs
    os.makedirs(os.path.dirname(args.train_output), exist_ok=True)
    os.makedirs(os.path.dirname(args.test_output), exist_ok=True)

    train_df.to_csv(args.train_output, index=False)
    test_df.to_csv(args.test_output, index=False)

    print(f"Wrote train to {args.train_output} with shape {train_df.shape}")
    print(f"Wrote test to {args.test_output} with shape {test_df.shape}")


if __name__ == '__main__':
    main()


