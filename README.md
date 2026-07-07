# Customer Churn Prediction: Local MLOps & DataOps Lifecycle Pipeline

This repository contains an end-to-end, automated machine learning application lifecycle designed to predict enterprise customer churn using a localized, high-velocity MLOps architecture. The system completely decouples processing from recording, replacing cloud-native logging structures (AWS Lambda and Amazon CloudWatch) with localized automation loops, an interactive **MLflow Tracking Server**, and programmatic Python API client architectures.

---

## 🏗️ Architectural Overview

The application is structured into three integrated functional sub-objectives:

1. **Sub-Objective 1 (Data Pipeline & DataOps):** Automated ingestion, median/mode missing-value imputation, categorical encoding, feature scaling, and statistical correlation tracking executed via a background scheduler every 2 minutes.

2. **Sub-Objective 2 (Machine Learning Pipeline & MLOps):** Structural 70/30 dataset partitioning and side-by-side metric tracking (Accuracy, Precision, Recall, F1-Score) of an ensemble Random Forest Classifier against a parametric Logistic Regression baseline.

3. **Sub-Objective 3 (Programmatic API Access):** Standalone API query utilities leveraging `MlflowClient()` primitives to extract key operational parameters directly from the tracking database without graphical interfaces.

---

## 📁 Repository Directory Structure

Ensure your workspace folders and files are arranged precisely as follows before testing:

```text
churn_project/
│
├── data/
│   └── customer_churn.csv       # Sourced Kaggle dataset (Ensure it contains records!)
│
├── pipeline_utils.py            # Automated preprocessing & EDA tracking logic
├── train.py                     # 70/30 Train/Test split & MLflow model tracking loops
├── query_metrics.py             # Programmatic API client retrieving run parameters
├── run_pipeline.sh              # Bash execution script harness for DataOps automation
└── README.md                    # This instruction documentation file
```

---

## 🚀 Step-by-Step Local Execution Methodology

Follow these steps directly inside your IDE terminal window (e.g., VS Code or PyCharm Terminal) to run, test, and verify your local application environment.

### Step 1: Environment Setup & Library Installation

Initialize your Python environment and install the complete analytical and tracking framework dependencies:

```bash
pip install mlflow scikit-learn pandas numpy
```

### Step 2: Seed the Local Training Workspace

Execute the primary machine learning pipeline script to ingest the raw source data, process the layout features, execute a 70/30 verification split, train both classification models, and generate your tracking metadata database locally:

```bash
python train.py
```

**Expected Output:** The console will display structural matrix dimensions along with computed classification parameters (Accuracy, Precision, Recall, F1-Score) for both evaluated algorithms.

### Step 3: Launch the Interactive MLOps Dashboard UI

Spin up the local MLflow tracking network server to visualize performance comparisons:

```bash
mlflow ui
```

* **Crucial Step Handling:** Your active terminal window will pause execution and display a server state listening line: `Listening at: http://127.0.0.1:5000`. Leave this terminal tab open.
* Open your internet browser and navigate directly to: **`http://127.0.0.1:5000`**
* Select both listed model execution runs inside the browser panel and click **Compare** to view your classification metrics side-by-side.

### Step 4: Programmatic API Interrogation (Sub-Objective 3)

Open a **new terminal tab or window** inside your IDE (leaving the Step 3 web server active in the background). Execute the standalone client query script to access run metadata using code:

```bash
python query_metrics.py
```

**Expected Output:** Bypassing all visual graphical boundaries, this script prints exactly four distinct application parameters directly onto your terminal grid using built-in API calls: Unique Run ID, Execution Flow Name, Local Source File Origin, and the final Metrics payload dictionary.

### Step 5: Activate the 2-Minute DataOps Automation Engine

To verify background workflow triggers, apply system permissions to your execution script and link it directly to your operating system's background daemon scheduler (`cron` on macOS/Linux):

1. Grant terminal file execution permissions:

   ```bash
   chmod +x run_pipeline.sh
   ```

2. Open your background schedule ledger grid file:

   ```bash
   crontab -e
   ```

3. Shift your terminal to insert text mode (press `i`), paste the following schedule mapping at the very bottom, and save/exit (`Esc`, then type `:wq` and press `Enter`):

   ```text
   */2 * * * * /path/to/churn_project/run_pipeline.sh
   ```

   > Replace `/path/to/churn_project/` with the absolute path to your local project directory.

4. **Verification:** Wait 2 to 4 minutes. A file titled **`pipeline_execution.log`** will be generated automatically in your folder workspace panel. Open it to check the continuous, sequential 2-minute workflow history tracking entries.

---

## 🛑 Clean System Environment Teardown

Once you have captured your required report screenshots, follow these steps to release your system resources and stop all active processes:

1. **Stop the Infinite DataOps Schedule:** Open your background ledger configuration by running `crontab -e`, erase your `*/2 * * * *` execution code line entirely, save, and exit.

2. **Kill Port 5000 Tracking Web Server:** To shut down the active MLflow web background processes, run this clean command in any terminal line:

   ```bash
   kill -9 $(lsof -t -i:5000)
   ```

3. **Wipe Local Logs (Optional):** If you wish to reset your workspace structure clean, you can safely delete the generated `mlruns/` directories and the text-based `pipeline_execution.log` file.
