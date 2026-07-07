# query_metrics.py
from mlflow.tracking import MlflowClient

def display_pipeline_api_details():
    client = MlflowClient()
    
    # Retrieve experiments through the built-in MLflow tracking API
    experiments = client.search_experiments()
    if not experiments:
        print("No active tracking metrics found. Please execute train.py first.")
        return
        
    # Get the latest experiment ID for Customer_Churn_Analysis
    exp_id = experiments[0].experiment_id
    runs = client.search_runs(experiment_ids=[exp_id])
    
    print("==========================================================")
    print(" PROGRAMMATIC API ACCESS: PIPELINE RUNTIME DETAILS       ")
    print("==========================================================\n")
    
    if not runs:
        print("No active tracking runs discovered inside this experiment.")
        return

    # Extract exactly 4 critical application runtime details (Activities 3.1 & 3.2)
    for i, run in enumerate(runs[:2]): # Fetch data from latest runs
        print(f"--- Application Run Artifact Target #{i+1} ---")
        
        # Detail 1: Unique Application Run Identifier
        print(f"1. Run ID (Deployment Context): {run.info.run_id}")
        
        # Detail 2: Run Name / Operational Flow Name
        print(f"2. Execution Flow Name:         {run.data.tags.get('mlflow.runName')}")
        
        # Detail 3: Safe Extraction of Local Source Script Origin from Tags
        source_name = run.data.tags.get('mlflow.source.name', 'Local Python Environment')
        print(f"3. Local Source Script Origin:  {source_name}")
        
        # Detail 4: Final Logged Application Performance Metric Dictionary
        print(f"4. Captured Runtime Metrics:    {run.data.metrics}")
        print("-" * 58)

if __name__ == "__main__":
    display_pipeline_api_details()