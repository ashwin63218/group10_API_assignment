# pipeline_utils.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def run_data_pipeline(data_path="data/customer_churn.csv"):
    print("=== [DataOps] Starting Data Pipeline Workflows ===")
    
    # 1.2 Data Ingestion
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please place your renamed file in the data/ folder.")
        
    df = pd.read_csv(data_path)
    print(f"[1.2 Ingestion] Loaded dataset successfully. Shape: {df.shape}")
    
    # 1.3 Data Pre-processing
    print("\n--- 1.3 Summary Statistics ---")
    print(df.describe(include='all'))
    
    print("\n--- 1.3 Data Types & Missing Values ---")
    print(df.dtypes)
    missing_counts = df.isnull().sum()
    print("Missing Values:\n", missing_counts)
    
    # Impute missing values (Numeric: Median, Categorical: Mode)
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
            
    # 1.4 Exploratory Data Analysis (EDA)
    # Drop unique string identifiers / high-cardinality text columns that can't be one-hot encoded easily
    cols_to_drop = ['Names', 'Onboard_date', 'Location', 'Company']
    print(f"\n[1.4 EDA] Dropping text metadata columns for training: {cols_to_drop}")
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    # Process any remaining categorical features if present
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        print(f"[1.4 EDA] Encoding remaining categorical columns: {categorical_cols}")
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Numeric correlation matrix calculation
    print("\n--- 1.4 Correlation Matrix (Top 5x5 Slice) ---")
    corr_matrix = df.corr()
    print(corr_matrix.iloc[:5, :5]) 
    
    # Feature scaling / Normalization
    X = df.drop(columns=['Churn']) 
    y = df['Churn']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_normalized = pd.DataFrame(X_scaled, columns=X.columns)
    
    print("\n=== [DataOps] Pipeline Run Successfully Completed ===")
    return X_normalized, y

if __name__ == "__main__":
    run_data_pipeline()