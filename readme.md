# 📊 Bank Client Churn Prediction

A comprehensive end-to-end data science project to predict client attrition (churn) for a bank, using real-world demographic, economic, and behavioral data. This project demonstrates advanced data cleaning, feature engineering, temporal data splitting to prevent leakage, and robust machine learning modeling.

---

## Executive Summary

This project predicts whether a bank client will churn, leveraging two real datasets. We built a full pipeline: data cleaning, fusion, EDA, feature engineering, and supervised ML (SVM, k-NN, Random Forest, Naive Bayes). A **temporal split** ensures realistic evaluation and prevents data leakage. The best model (SVM) achieves an **F1-score of 0.938** on the test set.

**Highlights:**
- Data fusion from two sources
- Handling missing data, outliers, and inconsistencies
- Feature engineering: discretization, encoding, scaling
- **Temporal split**: train on pre-2006, test on 2006+ clients
- Model evaluation: F1-score, accuracy, precision, recall

---

## Project Objectives

- **Understand the dataset:** Explore, clean, and preprocess raw banking data.
- **Feature engineering:** Handle missing values, outliers, categorical encodings, and scaling.
- **Data fusion:** Merge multiple data sources into a single, coherent dataset.
- **Modeling:** Train, validate, and test machine learning models to predict client attrition.
- **Analysis:** Interpret results with descriptive statistics and visualizations.

---

## 📁 Repository Structure

```
.
├── Banque_vfinal.ipynb       # Jupyter Notebook with full analysis
├── Application_BI.pdf        # Project report (French)
├── data/
│   ├── table1.csv            # Raw client data
│   └── table2.csv            # Additional client data
├── outputs/
│   ├── cleaned_data.csv      # Preprocessed dataset
│   ├── model.pkl             # Trained SVM model
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## Data Pipeline Overview

### 1️⃣ Data Loading
- Load `table1.csv` and `table2.csv`
- Inspect structure, types, and samples

### 2️⃣ Data Cleaning & Preprocessing
- Handle missing values (imputation/removal)
- Detect and treat outliers (IQR, Z-score)
- Standardize formats and codes
- Encode categorical variables (One-Hot, Ordinal)
- Scale features (Z-score, Min-Max)

### 3️⃣ Data Fusion
- Merge on: `Sexe`, `SituationFamiliale`, `ClientTypeCode`, `DateAdhesion`, `StatutSocietaireCode`, `MotifDemissionCode`, `RevenuMontant`, `NombreEnfants`, `DateDemission`
- Validate merged dataset

### 4️⃣ Feature Engineering
- Discretize age and income
- Dimensionality reduction (PCA)
- Balance classes (oversampling/undersampling if needed)

### 5️⃣ Modeling
- **Temporal split:** Train on pre-2006 churners/non-churners, test on 2006 churners and 2007 actives
- Train SVM, k-NN, Random Forest, Naive Bayes
- Evaluate with F1-score, accuracy, precision, recall

---

## Key Insights from EDA

- Distribution of categorical and numerical features
- Correlations and relationships across attributes
- Impact of demographic factors on churn

---

## 🤖 Model Results

- **Best Model:** SVM (C=10, RBF kernel)
- **F1-Score:** 0.938
- **Accuracy:** 0.950
- **Recall:** 0.925
- **Validation:** 5-fold cross-validation

---

## Tech Stack

| Tool/Library | Purpose                          |
| ------------ | -------------------------------- |
| Python       | Programming language             |
| Pandas       | Data manipulation                |
| Numpy        | Numerical computations           |
| Matplotlib   | Data visualization               |
| Seaborn      | Enhanced visualizations          |
| Scikit-learn | ML algorithms & preprocessing    |
| SciPy        | Statistical analysis             |
| Jupyter      | Interactive notebook environment |

---

## 🏗️ Dependencies

Install with:

```bash
pip install -r requirements.txt
```

Typical dependencies:

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
scipy
```

---

## How to Run

1. **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/banque-attrition-prediction.git
    cd banque-attrition-prediction
    ```
2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
3. **Run the notebook**
    ```bash
    jupyter notebook Banque_vfinal.ipynb
    ```
4. **Follow the notebook step by step.**

---

##  Next Steps

- Test additional ML models (e.g., XGBoost, LightGBM)
- Explore feature selection techniques
- Enhance explainability with SHAP or LIME
- Deploy as a web app for real-time churn prediction

---

## License

This project is released under the MIT License.

---

## 📬 Contact

Team member:

- **Paul Moïse GANGBADJA**
- **Fataï IDRISSOU**


---

✅ **[Full Report PDF](./Application_BI.pdf)**  
✅ **[Jupyter Notebook](./Banque_vfinal.ipynb)**

