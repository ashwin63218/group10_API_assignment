# train.py
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Ingest data pipeline steps from Sub-Objective 1
from pipeline_utils import run_data_pipeline

def main():
    # Set the experiment name in MLflow
    mlflow.set_experiment("Customer_Churn_Analysis")
    
    # Retrieve preprocessed features and labels
    X, y = run_data_pipeline()
    
    # 2.2 Model Training: Strict 70% Train and 30% Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    print(f"\n[2.2 Split] Training Matrix: {X_train.shape}, Testing Matrix: {X_test.shape}")
    
    # --- Algorithm 1: Random Forest ---
    with mlflow.start_run(run_name="Random_Forest_Classifier"):
        print("\nTraining Random Forest Classifier...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        predictions = rf.predict(X_test)
        
        # Calculate the 4 required MLOps metrics
        acc = accuracy_score(y_test, predictions)
        prec = precision_score(y_test, predictions, zero_division=0)
        rec = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        
        # MLOps tracking logs
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(rf, "model")
        print(f"Logged RF Metrics -> Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

    # --- Algorithm 2: Logistic Regression ---
    with mlflow.start_run(run_name="Logistic_Regression"):
        print("\nTraining Logistic Regression...")
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        predictions = lr.predict(X_test)
        
        # Calculate the 4 required MLOps metrics
        acc = accuracy_score(y_test, predictions)
        prec = precision_score(y_test, predictions, zero_division=0)
        rec = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        
        # MLOps tracking logs
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(lr, "model")
        print(f"Logged LR Metrics -> Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

if __name__ == "__main__":
    main()