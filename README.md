# MACHINE LEARNING BASED SCHOLARSHIP ELIGIBILITY PREDICTION

## Project Overview

**Machine Learning Based Scholarship Eligibility Prediction** is a machine learning classification project that analyzes scholarship-related data and predicts the **Outcome** for a given record.

The project uses a dataset containing **245,760 records and 12 attributes**. Eleven attributes are used as feature variables, while **Outcome** is used as the target variable.

The project implements and compares two supervised machine learning algorithms:

* Decision Tree Classifier
* Random Forest Classifier

The models are trained using an **80:20 stratified train-test split** with `random_state = 42`. Categorical variables are processed using **One-Hot Encoding**.

---

## Problem Statement

Scholarship-related outcomes may depend on multiple attributes such as education qualification, gender, community, religion, ex-servicemen status, disability, sports status, annual percentage, income, India-related eligibility category, and scholarship name.

Manually analysing these conditions across a large dataset can be difficult. Therefore, this project aims to develop a machine learning classification model capable of learning patterns from scholarship-related data and predicting the **Outcome** for a given record.

> **Objective:** To develop a machine learning classification model that can learn patterns from scholarship-related data and predict the Outcome for a given record.

---

## Dataset

**Dataset File:** `dataset_combined.xlsx`
**Sheet:** `dataset_revised`
**Source:** Kaggle

### Dataset Statistics

| Property          |      Value |
| ----------------- | ---------: |
| Total Records     |    245,760 |
| Total Attributes  |         12 |
| Feature Variables |         11 |
| Target Variable   |    Outcome |
| Missing Values    |          0 |
| Training Data     |        80% |
| Testing Data      |        20% |
| Split Type        | Stratified |
| Random State      |         42 |

### Attributes

| Attribute               | Description                        |
| ----------------------- | ---------------------------------- |
| Name                    | Scholarship name                   |
| Education Qualification | Education qualification/category   |
| Gender                  | Gender category                    |
| Community               | Community category                 |
| Religion                | Religion category                  |
| Exservice-men           | Ex-servicemen status               |
| Disability              | Disability status                  |
| Sports                  | Sports status                      |
| Annual-Percentage       | Annual percentage category         |
| Income                  | Income category                    |
| India                   | India-related eligibility category |
| Outcome                 | Target variable                    |

### Target Distribution

|   Outcome |     Records | Percentage |
| --------: | ----------: | ---------: |
|         0 |     212,992 |   86.6667% |
|         1 |      32,768 |   13.3333% |
| **Total** | **245,760** |   **100%** |

The target variable is imbalanced, with Outcome 0 representing **86.6667%** of the dataset and Outcome 1 representing **13.3333%**.

---

## Machine Learning Algorithms

### 1. Decision Tree Classifier

Decision Tree is a supervised classification algorithm that learns decision rules from input features and predicts the target class.

The implementation uses:

```python
DecisionTreeClassifier(random_state=42)
```

Decision Tree is suitable for this project because the dataset contains structured categorical attributes that can be converted into numerical representations using encoding.

### 2. Random Forest Classifier

Random Forest is an ensemble classification algorithm that combines multiple decision trees and uses their combined predictions to classify observations.

The implementation uses:

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
```

Random Forest is suitable because it provides an ensemble-based approach for classification using multiple decision trees.

---

## Methodology

The complete machine learning workflow is:

```text
Dataset
   ↓
Data Loading
   ↓
Data Inspection
   ↓
Feature and Target Separation
   ↓
Categorical Encoding
   ↓
Stratified Train-Test Split
   ↓
Decision Tree Training
   ↓
Random Forest Training
   ↓
Predictions
   ↓
Confusion Matrix
   ↓
Accuracy
   ↓
Precision
   ↓
Recall
   ↓
F1-Score
   ↓
Algorithm Comparison
```

### Experimental Configuration

| Parameter                | Setting               |
| ------------------------ | --------------------- |
| Dataset                  | dataset_combined.xlsx |
| Sheet                    | dataset_revised       |
| Training Size            | 80%                   |
| Testing Size             | 20%                   |
| Random State             | 42                    |
| Stratification           | Yes                   |
| Encoding                 | One-Hot Encoding      |
| Algorithm 1              | Decision Tree         |
| Algorithm 2              | Random Forest         |
| Random Forest Estimators | 100                   |
| Feature Scaling          | Not used              |

StandardScaler is not used because Decision Tree and Random Forest are tree-based algorithms and do not require feature scaling.

---

## Implementation

The implementation is developed using **Python, Pandas and Scikit-learn**.

### Required Libraries

```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
```

### Dataset Loading

```python
df = pd.read_excel(
    "dataset_combined.xlsx",
    sheet_name="dataset_revised"
)
```

### Feature and Target Separation

```python
X = df.drop(columns=["Outcome"])
y = df["Outcome"]
```

### Categorical Encoding

```python
categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns

numeric_features = X.select_dtypes(
    exclude=["object", "category"]
).columns

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)
```

### Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

### Decision Tree

```python
decision_tree = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            DecisionTreeClassifier(random_state=42)
        )
    ]
)

decision_tree.fit(X_train, y_train)

y_pred_dt = decision_tree.predict(X_test)
```

### Random Forest

```python
random_forest = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

random_forest.fit(X_train, y_train)

y_pred_rf = random_forest.predict(X_test)
```

### Evaluation

```python
cm_dt = confusion_matrix(y_test, y_pred_dt)

accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)

cm_rf = confusion_matrix(y_test, y_pred_rf)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
```

---

## Results

Both algorithms achieved identical performance on the selected test dataset.

### Performance Comparison

| Metric    | Decision Tree | Random Forest |
| --------- | ------------: | ------------: |
| Accuracy  |          100% |          100% |
| Precision |          100% |          100% |
| Recall    |          100% |          100% |
| F1-Score  |          100% |          100% |

### Confusion Matrix — Decision Tree

| Actual / Predicted |      0 |     1 |
| ------------------ | -----: | ----: |
| 0                  | 42,598 |     0 |
| 1                  |      0 | 6,554 |

### Confusion Matrix — Random Forest

| Actual / Predicted |      0 |     1 |
| ------------------ | -----: | ----: |
| 0                  | 42,598 |     0 |
| 1                  |      0 | 6,554 |

Both models correctly classified all **49,152 test records** in this experiment.

---

## Performance Evaluation

The models were evaluated using the following metrics.

### Accuracy

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

For both models:

```text
= (6,554 + 42,598) / (6,554 + 42,598 + 0 + 0)
= 49,152 / 49,152
= 1.0000
= 100%
```

### Precision

```text
Precision = TP / (TP + FP)
```

For both models:

```text
= 6,554 / (6,554 + 0)
= 1.0000
= 100%
```

### Recall

```text
Recall = TP / (TP + FN)
```

For both models:

```text
= 6,554 / (6,554 + 0)
= 1.0000
= 100%
```

### F1-Score

```text
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
```

For both models:

```text
= 2 × (1 × 1) / (1 + 1)
= 1.0000
= 100%
```

---

## Conclusion

The project successfully implemented a machine learning classification workflow for scholarship-related data containing **245,760 records and 12 attributes**.

Eleven attributes were used as features and **Outcome** was used as the target variable. Categorical attributes were processed using **One-Hot Encoding**, followed by an **80:20 stratified train-test split** with `random_state=42`.

Two classification algorithms were implemented:

1. Decision Tree Classifier
2. Random Forest Classifier

Both algorithms achieved:

* **100% Accuracy**
* **100% Precision**
* **100% Recall**
* **100% F1-Score**

Both models also produced the same confusion matrix:

```text
[[42598, 0],
 [0, 6554]]
```

Based on the evaluated metrics, **both algorithms demonstrated identical performance on the test dataset**.

---

## Project Structure

```text
MACHINE-LEARNING-SCHOLARSHIP-PREDICTION/
│
├── dataset_combined.xlsx
│
├── scholarship_prediction.py
│
├── README.md
│
└── screenshots/
    ├── dataset_loading.png
    ├── dataset_inspection.png
    ├── preprocessing_encoding.png
    ├── train_test_split.png
    ├── decision_tree_training.png
    ├── random_forest_training.png
    ├── confusion_matrix.png
    └── final_evaluation.png
```

---

## Technologies Used

* **Python**
* **Pandas**
* **Scikit-learn**
* **One-Hot Encoding**
* **Decision Tree**
* **Random Forest**
* **Jupyter Notebook / VS Code**
* **Microsoft Excel (.xlsx)**

---

## Key Highlights

* 245,760 scholarship-related records analyzed
* 12 attributes
* 11 feature variables
* 1 target variable
* 0 missing values
* One-Hot Encoding
* Stratified 80:20 train-test split
* Decision Tree Classifier
* Random Forest Classifier
* 100 Random Forest estimators
* Confusion Matrix evaluation
* Accuracy, Precision, Recall and F1-Score evaluation
* 100% measured performance for both models on the selected test set

---

## References

1. Kaggle – Dataset Source
2. Python Documentation
3. Pandas Documentation
4. Scikit-learn Documentation
5. Scikit-learn DecisionTreeClassifier Documentation
6. Scikit-learn RandomForestClassifier Documentation
7. Scikit-learn Metrics Documentation
