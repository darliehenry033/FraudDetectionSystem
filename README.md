# Credit Card Fraud Detection Pipeline

An end-to-end machine learning workflow designed to identify fraudulent financial transactions. This project focuses heavily on robust data preprocessing, exploratory data analysis (EDA), and strategic handling of highly imbalanced anomaly detection data.

##  Project Overview

Credit card fraud costs financial institutions billions of dollars annually. This repository demonstrates a data science approach to predicting fraudulent transactions using a dataset of **284,807 transactions**.

The primary challenge of this project is the extreme class imbalance: **only 0.17% of the total transactions are labeled as fraud**, requiring specialized preprocessing and evaluation strategies beyond standard accuracy metrics.

---

##  Tech Stack & Libraries

* **Language:** Python
* **Data Manipulation:** `pandas`, `numpy`
* **Data Visualization:** `seaborn`, `matplotlib`
* **Machine Learning:** `scikit-learn` *(implied pipeline targets)*

---

##  Data Pipeline Architecture

### 1. Data Cleaning & Structural Auditing

Before diving into modeling, a strict structural check was enforced on the dataset:

* **Whitespace Sanitization:** Stripped implicit trailing or leading spaces from all column headers to prevent key errors during indexing.
* **Null-Value Verification:** Confirmed zero missing values across all 31 features, ensuring data integrity from the start.
* **Deduplication:** Identified and eliminated **1,081 duplicate entries**, shrinking the dataset size from 284,807 records down to a high-fidelity pool of **283,726 unique transactions**.

### 2. Dataset Characteristics & Insights

* **Features $V1$ through $V28$:** Numerical predictors obtained via a PCA transformation due to privacy constraints.
* **Time & Amount:** The only unscaled features in the raw data.
* **Target Distribution ($Class$):**
* `0`: Legitimate transactions
* `1`: Fraudulent transactions (Anomalies representing roughly **0.173%** of the dataset)



>  **Hiring Manager Note:** Because accuracy will comfortably exceed 99% simply by guessing "not fraud," this project utilizes advanced metrics like **Precision-Recall AUC (AUPRC)** and **Recall** rather than raw classification accuracy to evaluate model performance.

### 3. Outlier Mitigation & Noise Control

* Visualized spatial distributions and variance using boxplots and distribution curves to cautiously isolate extreme operational noise without deleting vital fraud signals.

---

##  Getting Started

### Prerequisites

Ensure you have Python installed, then set up the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn

```

### Execution

1. Clone this repository to your local machine.
2. Place your raw `creditcard.csv` file inside the root directory.
3. Execute the analysis script or open the Jupyter Notebook:

```bash
jupyter notebook credit_card_fraud.ipynb

```

---

##  Future Scope & Extensions

* [ ] Implement **SMOTE** (Synthetic Minority Over-sampling Technique) or class weight balances to handle severe skewness.
* [ ] Compare performance across **Isolation Forests**, **Random Forests**, and **XGBoost**.
* [ ] Wrap the final optimal model into a lightweight **FastAPI** or Flask application for real-time inference testing.
