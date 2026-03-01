# MSMARCO TF-IDF MLlib Pipeline  
End-to-End Scalable Text Classification Pipeline using Apache Spark

## 📌 Project Overview

This project implements a scalable end-to-end text classification pipeline on the MSMARCO dataset using Apache Spark MLlib.  

It demonstrates:

- Large-scale data ingestion (TSV → Parquet)
- TF-IDF feature engineering
- Distributed model training
- Cross-validation tuning
- Scalability benchmarking
- Data quality reporting
- Tableau-ready metric exports
- Comparison with scikit-learn baseline

The pipeline is designed to simulate real-world big data ML workflows on distributed systems.

---

## 🏗 Architecture

Raw Data (TSV)
        ↓
Spark Ingestion
        ↓
Parquet Dataset
        ↓
TF-IDF Feature Engineering
        ↓
Train/Test Split
        ↓
Model Training (LR, SVM, NB)
        ↓
Cross Validation
        ↓
Evaluation Metrics
        ↓
Tableau Visualization

---

## 📂 Project Structure
msmarco-tfidf-mllib-pipeline/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/                           # Core pipeline scripts
│   ├── 1_ingest_to_parquet.py
│   ├── 2_feature_engineering.py
│   ├── 3_model_training.py
│   ├── 4_sklearn_baseline.py
│   ├── 5_crossval_tuning.py
│   ├── 6_scalability_experiment.py
│   └── 7_data_quality_export.py
│
├── notebooks/                     # (Optional exploration)
│   └── sample.py
│
├── data/
│   ├── raw/                       # (NOT pushed to GitHub)
│   ├── processed/                 # (Ignored in GitHub)
│   └── samples/
│
├── outputs/
│   ├── metrics_tableau/
│   ├── scalability_results.csv
│   └── tableau_data_quality/
│
├── models/                        # Saved trained models
│
└── tabulae/                       # Tableau dashboards
    └── Naga jaswanth.twbx


---

## 🚀 Key Components

### 1️⃣ Data Ingestion
- Converts raw MSMARCO TSV files to partitioned Parquet
- Generates positive/negative query-document pairs
- Saves sample dataset for experimentation

### 2️⃣ Feature Engineering
- Regex Tokenization
- Stopword Removal
- HashingTF
- IDF weighting
- Train/Test split
- Saves trained TF-IDF pipeline

### 3️⃣ Distributed Model Training
Models trained using Spark MLlib:
- Logistic Regression
- Linear SVC
- Naive Bayes

Metrics:
- Accuracy
- F1 Score
- Precision
- Recall
- ROC-AUC
- Confusion Matrix

Metrics exported to CSV for Tableau dashboards.

### 4️⃣ Cross Validation
- 3-fold cross-validation
- Hyperparameter grid search
- Parallelized model tuning

### 5️⃣ Scalability Experiment
Evaluates:
- Shuffle partitions impact
- Dataset size scaling
- Training time vs performance tradeoff

Outputs benchmarking CSV.

### 6️⃣ Data Quality Reporting
Exports:
- Missing value statistics
- Label distribution
- Text length statistics

Designed for Tableau dashboards.

### 7️⃣ Sklearn Baseline
- Logistic Regression using scikit-learn
- Compares distributed Spark performance with local ML baseline


## 🛠 Tech Stack

- Python
- Apache Spark (PySpark)
- Spark MLlib
- scikit-learn
- NumPy
- Tableau (for visualization)

---

## 💡 Skills Demonstrated

- Distributed Data Processing
- Feature Engineering for NLP
- ML Model Evaluation
- Cross Validation at Scale
- Performance Benchmarking
- Data Engineering Best Practices
- Production-style Project Structure

---

## ▶️ How to Run

### Step 1: Ingest Data

python 1_ingest_to_parquet.py


### Step 2: Feature Engineering

python 2_feature_engineering.py


### Step 3: Train Models

python 3_model_training.py


### Step 4: Cross Validation

python 5_crossval_tuning.py


### Step 5: Scalability Benchmark

python 6_scalability_experiment.py


---

## 📈 Future Improvements

- Add Docker containerization
- Deploy on Spark cluster (YARN/Databricks)
- Add Airflow orchestration
- Integrate MLflow tracking

---
---