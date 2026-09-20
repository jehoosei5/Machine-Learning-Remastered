# Data Mining Capstone — Jehoshaphat Osei Boateng

Capstone work for **Data Mining & Information Visualization**, covering three problem types:

| Project | Type | Notebook |
|---------|------|----------|
| House Price Prediction | Supervised **regression** | `House_Price_Regression_Capstone.ipynb` |
| Mall Customer Segmentation | Unsupervised **clustering** | `Mall_Customers_Clustering_Capstone.ipynb` |
| Telco Customer Churn | Supervised **classification** | `Telco_Churn_Classification_Capstone.ipynb` |

Each notebook follows the course workflow: data selection → understanding → preprocessing → feature selection → modeling → validation/testing → evaluation → visualization → interpretation.

---

## Repository contents

```
Capstone/
├── House_Price_Regression_Capstone.ipynb
├── Mall_Customers_Clustering_Capstone.ipynb
├── Telco_Churn_Classification_Capstone.ipynb
├── train_cleaned.csv              # House prices (cleaned train)
├── test_cleaned.csv               # House prices (cleaned Kaggle test)
├── Mall_Customers.csv             # Mall customers (raw)
├── mall_customers_clusters.csv    # Mall customers + cluster labels
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── README.md
```

---

## Requirements

- Python 3.9+
- Jupyter Notebook or JupyterLab

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy imbalanced-learn
```

| Package | Used for |
|---------|----------|
| `numpy`, `pandas` | Data handling |
| `matplotlib`, `seaborn` | Visualization |
| `scikit-learn` | Preprocessing, models, metrics |
| `scipy` | Hierarchical clustering dendrogram |
| `imbalanced-learn` | SMOTE (Telco churn only) |

---

## How to run

1. Open a terminal in this folder (`Capstone/`).
2. Install the packages above (or use an Anaconda environment that already has them).
3. Launch Jupyter and open any of the three notebooks:

```bash
jupyter notebook
```

4. Run all cells top to bottom. CSV paths are relative to this directory, so keep the notebooks and data files together.

---

## 1. House Price Prediction (Regression)

**Goal:** Predict continuous `SalePrice` for residential homes.

**Data:** [Kaggle House Prices — Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)  
- `train_cleaned.csv` — 1,460 homes, 81 columns (features + `SalePrice`)  
- `test_cleaned.csv` — 1,459 homes without labels (optional Kaggle-style inference)

**Pipeline highlights:**
- EDA on target skew, correlations, and neighborhood effects
- Median / mode imputation, `StandardScaler`, `OneHotEncoder`
- Feature selection with `SelectKBest` (`f_regression`)
- Train / validation / test split (~70% / 15% / 15%)
- Models: Linear Regression, Ridge, Decision Tree, Random Forest, Gradient Boosting
- Metrics: **MAE**, **RMSE**, **R²** (best model chosen by validation RMSE)
- Plots: actual vs predicted, residuals, feature importance

**Typical drivers:** overall quality, living area, garage/basement size, neighborhood.

---

## 2. Mall Customer Segmentation (Clustering)

**Goal:** Discover natural shopper segments with no labeled target.

**Data:**
- `Mall_Customers.csv` — 200 customers (`Gender`, `Age`, `Annual Income (k$)`, `Spending Score (1-100)`)
- `mall_customers_clusters.csv` — same records with assigned cluster labels

**Pipeline highlights:**
- Drop `CustomerID`; encode gender for profiling
- Primary features: annual income and spending score (age optional)
- `StandardScaler` before distance-based clustering
- Algorithms: **K-Means** and **Agglomerative Hierarchical** (Ward)
- Choose **k** with elbow method, silhouette score, and dendrogram (typically **k = 5**)
- Evaluation: silhouette, Davies–Bouldin, Calinski–Harabasz
- Cluster profiles mapped to marketing actions (premium loyalty, upsell, value bundles, etc.)

---

## 3. Telco Customer Churn (Classification)

**Goal:** Predict whether a customer will churn (`Yes` / `No`).

**Data:** `WA_Fn-UseC_-Telco-Customer-Churn.csv` — 7,043 customers, 21 columns.

**Pipeline highlights:**
- Drop `customerID`; coerce `TotalCharges` to numeric
- Stratified train / validation / test split (~70% / 15% / 15%)
- Imputation, scaling, one-hot encoding
- **SMOTE** on the training fold only (class imbalance ~73% / 27%)
- Feature selection with mutual information (`SelectKBest`)
- Models: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
- Emphasis on **Recall**, **F1**, and **ROC-AUC** (not accuracy alone)
- Plots: confusion matrix, ROC, precision–recall, feature importance

**Typical drivers:** contract type, tenure, internet service, payment method, charges.

---

## Shared design choices

- Fixed `RANDOM_STATE = 42` for reproducibility
- Preprocessors fit on training data only to avoid leakage
- Validation used for model selection; held-out test for final reporting
- Visualizations and written interpretation included in each notebook

---

## Author

**Jehoshaphat Osei Boateng**  
Course: Data Mining & Information Visualization — Capstone
