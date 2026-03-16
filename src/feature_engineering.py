import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw Telco churn dataframe.
    - Convert TotalCharges to numeric
    - Drop customerID
    - Fill missing values
    """
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    df.drop(columns=['customerID'], inplace=True, errors='ignore')
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode binary and multi-class categorical columns.
    """
    df = df.copy()

    # Binary columns: map Yes/No to 1/0
    binary_cols = [
        'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
        'Churn', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0,
                                   'No phone service': 0,
                                   'No internet service': 0})

    # Gender encoding
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

    # One-hot encode remaining categoricals
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features:
    - tenure_group: bucket tenure into ranges
    - charges_per_month_ratio: TotalCharges / tenure ratio
    - is_long_term: binary flag for tenure > 24 months
    """
    df = df.copy()

    # Tenure grouping
    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, 60, 72],
        labels=['0-1yr', '1-2yr', '2-4yr', '4-5yr', '5+yr']
    )
    df = pd.get_dummies(df, columns=['tenure_group'], drop_first=True)

    # Charge ratio (avoid divide by zero)
    df['charges_per_month_ratio'] = df['TotalCharges'] / (df['tenure'] + 1)

    # Long-term customer flag
    df['is_long_term'] = (df['tenure'] > 24).astype(int)

    return df


def scale_numeric(df: pd.DataFrame,
                  num_cols: list,
                  scaler=None):
    """
    Scale numeric columns. If scaler is None, fit a new StandardScaler.
    Returns (scaled_df, fitted_scaler).
    """
    df = df.copy()
    if scaler is None:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        df[num_cols] = scaler.transform(df[num_cols])
    return df, scaler
