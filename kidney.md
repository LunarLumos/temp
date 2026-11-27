# Kidney Disease Notebook: cells to copy into Kaggle

Add the cells in this order. Each block below is one cell: the heading says whether to add a **Markdown** cell or a **Code** cell, and the box holds exactly what to paste.

## Cell 1: Markdown

````markdown
# Medical Screening Expert System for Chronic Kidney Disease
CIS313 Semester Assignment, Topic #11. Dataset: Chronic Kidney Disease (UCI Machine Learning Repository).
````

## Cell 2: Code

```python
# Information

name = "A. Islam (Aadil)"
student_id = "242-16-025"
section = "20_A"
project_title = "Medical Screening Expert System for Chronic Kidney Disease"

print("===================================")
print("        PROJECT INFORMATION")
print("===================================")
print(f"Name          : {name}")
print(f"ID            : {student_id}")
print(f"Section       : {section}")
print(f"Project Title : {project_title}")
print("===================================")
```

## Cell 3: Markdown

````markdown
## 1. Import Libraries
````

## Cell 4: Code

```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix)
from sklearn.inspection import permutation_importance

sns.set_theme(style="whitegrid")
```

## Cell 5: Markdown

````markdown
## 2. Load Dataset (Q3)
````

## Cell 6: Code

```python
file_path = "data/chronic_kidney_disease.csv"

# If the file is not on this computer (for example on Kaggle or Colab),
# read the same file directly from the UCI website
if not os.path.exists(file_path):
    file_path = "https://archive.ics.uci.edu/static/public/336/data.csv"

df = pd.read_csv(file_path)

df.head()
```

## Cell 7: Markdown

````markdown
## 3. Explore Dataset (Q3)
````

## Cell 8: Code

```python
print("Shape:", df.shape)
df.info()
```

## Cell 9: Code

```python
# Feature summary table
df.describe().round(2)
```

## Cell 10: Markdown

````markdown
`random_state` fixes the "random" choice so the same rows are picked every time the notebook runs.
````

## Cell 11: Code

```python
df.sample(5, random_state=42)
```

## Cell 12: Markdown

````markdown
*(Write your interpretation here.)*
````

## Cell 13: Markdown

````markdown
## 4. Data Preprocessing (Q3)
### 4.1 Clean the text values
````

## Cell 14: Markdown

````markdown
Some text values have a hidden tab character (for example `"ckd\t"` and `"\tno"`), so the same answer looks like two different values. `str.strip()` removes the extra spaces and tabs.
````

## Cell 15: Code

```python
# Columns that hold text (the other columns hold numbers)
text_columns = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane", "class"]
numeric_columns = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot",
                   "hemo", "pcv", "wbcc", "rbcc"]

print("Before cleaning:", df["class"].unique())

for col in text_columns:
    df[col] = df[col].str.strip()

print("After cleaning :", df["class"].unique())
```

## Cell 16: Markdown

````markdown
*(Write your interpretation here.)*
````

## Cell 17: Markdown

````markdown
### 4.2 Missing values and duplicates
````

## Cell 18: Code

```python
# Check for missing values and duplicate rows
print("Missing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("Rows with no missing value at all:", len(df.dropna()))
```

## Cell 19: Code

```python
# Chart: number of missing values in each column
missing_counts = df.isnull().sum().sort_values(ascending=False)

plt.figure(figsize=(12, 4))
sns.barplot(x=missing_counts.index, y=missing_counts.values, color="steelblue")
plt.title("Missing Values per Column")
plt.xlabel("Column")
plt.ylabel("Number of missing values")
plt.xticks(rotation=45)
plt.show()
```

## Cell 20: Markdown

````markdown
Removing every row with a missing value would throw away most of the patients, so the missing values are filled instead: the median for number columns and the most common value (mode) for text columns.
````

## Cell 21: Code

```python
# Number columns: fill with the median of the column
for col in numeric_columns:
    df[col] = df[col].fillna(df[col].median())

# Text columns: fill with the most common value of the column
for col in text_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

print("Missing values left:", df.isnull().sum().sum())
```

## Cell 22: Markdown

````markdown
*(Write your interpretation here.)*
````

## Cell 23: Markdown

````markdown
### 4.3 Create the target
The `class` column is text. It is changed to numbers: 1 = chronic kidney disease (ckd), 0 = no chronic kidney disease (notckd).
````

## Cell 24: Code

```python
df["target"] = (df["class"] == "ckd").astype(int)

# The original column is no longer needed (keeping it would give the answer away)
df = df.drop(columns="class")

df["target"].value_counts()
```

## Cell 25: Markdown

````markdown
*(Write your interpretation here.)*
````

## Cell 26: Markdown

````markdown
## 5. Data Visualization (Q3)
### Chart 1: Class distribution
````

## Cell 27: Code

```python
# Number of patients in each class
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="target")
plt.title("Class Distribution")
plt.xlabel("Target (0 = no kidney disease, 1 = kidney disease)")
plt.ylabel("Number of patients")
plt.show()
```

## Cell 28: Markdown

````markdown
*(Write your interpretation here.)*
````

## Cell 29: Markdown

````markdown
### Chart 2: Correlation heatmap
````

## Cell 30: Code

```python
# Correlation between the number columns and the target
plt.figure(figsize=(13, 10))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Between Numeric Features and Target")
plt.show()
```

## Cell 31: Code

```python
# Correlation of each numeric feature with the target, from highest to lowest
df.corr(numeric_only=True)["target"].drop("target").sort_values(ascending=False).round(2)
```

## Cell 32: Markdown

````markdown
*(Write your interpretation here.)*
````

## Cell 33: Markdown

````markdown
### Chart 3: Hemoglobin and serum creatinine by target
````

## Cell 34: Code

```python
# How two blood test results differ between the two classes
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, col in zip(axes, ["hemo", "sc"]):
    sns.boxplot(data=df, x="target", y=col, ax=ax)
    ax.set_title(f"{col} by target")
    ax.set_xlabel("Target (0 = no kidney disease, 1 = kidney disease)")

plt.tight_layout()
plt.show()
```

## Cell 35: Code

```python
# Average hemoglobin and serum creatinine in each class
df.groupby("target")[["hemo", "sc"]].mean().round(2)
```

## Cell 36: Markdown

````markdown
*(Write your interpretation here.)*
````

## Cell 37: Markdown

````markdown
## 6. Encode, Split and Scale (Q3)
````

## Cell 38: Markdown

````markdown
The text columns are changed to 0/1 columns with one-hot encoding. Every text column has only two values (for example yes/no), so one 0/1 column is enough for each (`drop_first=True`).
````

## Cell 39: Code

```python
# All text columns except the target
categorical_columns = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]

df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True, dtype=int)

df_encoded.head()
```

## Cell 40: Code

```python
X = df_encoded.drop(columns="target")
y = df_encoded["target"]

print("X shape:", X.shape)
print("y shape:", y.shape)
```

## Cell 41: Markdown

````markdown
80% of the data is used for training and 20% for testing.
`stratify=y` keeps the same proportion of each class in both sets.
````

## Cell 42: Code

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set:", X_train.shape)
print("Testing set :", X_test.shape)
```

## Cell 43: Code

```python
# Keep the original (unscaled) age of the test patients for Section 11
age_test = X_test["age"]
```

## Cell 44: Code

```python
# Scale the features (fit only on training data to avoid data leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## Cell 45: Markdown

````markdown
*(Write your interpretation here.)*
````

## Cell 46: Markdown

````markdown
## 7. Train Models (Q4)
````

## Cell 47: Markdown

````markdown
Five different models are trained on the same training data:

| Model | How it works | Main hyperparameter |
|---|---|---|
| Logistic Regression | Weighted sum of the features, turned into a probability | `C` |
| KNN | Looks at the K most similar patients and takes a vote | `n_neighbors` |
| Decision Tree | Asks a chain of yes/no questions, like the rules of an expert system | `max_depth` |
| Naive Bayes | Calculates the probability of each class and picks the larger one | `var_smoothing` |
| Random Forest | Many decision trees voting together | `n_estimators` |
````

## Cell 48: Code

```python
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    print(f"{name} trained")
```

## Cell 49: Markdown

````markdown
### Rules of the expert system
A decision tree is a set of IF-THEN rules. The same tree is trained here on the unscaled values, so the rules show real medical units (class 1 = kidney disease, class 0 = no kidney disease).
````

## Cell 50: Code

```python
rule_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
rule_tree.fit(X_train, y_train)

print(export_text(rule_tree, feature_names=list(X.columns)))
```

## Cell 51: Markdown

````markdown
*(Write your interpretation here.)*
````

## Cell 52: Markdown

````markdown
## 8. Model Outputs (Q5)
````

## Cell 53: Markdown

````markdown
A large gap between training accuracy and test accuracy is a sign of overfitting.
````

## Cell 54: Code

```python
# Predictions of every model on the test set
predictions = {}

for name, model in models.items():
    predictions[name] = model.predict(X_test_scaled)

# Training accuracy vs test accuracy
accuracy_rows = []

for name, model in models.items():
    train_pred = model.predict(X_train_scaled)

    accuracy_rows.append({
        "Model": name,
        "Training Accuracy": accuracy_score(y_train, train_pred),
        "Test Accuracy": accuracy_score(y_test, predictions[name]),
    })

pd.DataFrame(accuracy_rows).round(3)
```

## Cell 55: Code

```python
# Confusion matrix of every model
class_names = ["No CKD", "CKD"]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for ax, name in zip(axes, models):
    cm = confusion_matrix(y_test, predictions[name])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

# There are 5 models, so the 6th box is left empty
axes[5].axis("off")

plt.tight_layout()
plt.show()
```

## Cell 56: Code

```python
# Classification report of every model
for name in models:
    print("=====", name, "=====")
    print(classification_report(y_test, predictions[name], target_names=class_names))
```

## Cell 57: Markdown

````markdown
*(Write your interpretation here.)*
````

## Cell 58: Markdown

````markdown
## 9. Evaluate and Compare Models (Q6)
````

## Cell 59: Markdown

````markdown
ROC-AUC needs the predicted probability of class 1 (kidney disease), not the 0/1 prediction, so `predict_proba(...)[:, 1]` is used for it.
````

## Cell 60: Code

```python
results = []

for name, model in models.items():
    y_pred = predictions[name]
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    })

results_df = pd.DataFrame(results).sort_values(by=["Accuracy", "F1 Score"], ascending=False)
results_df.round(3).reset_index(drop=True)
```

## Cell 61: Code

```python
# Model comparison chart
results_df.plot.bar(x="Model", y=["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
                    figsize=(12, 5))
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.xlabel("")
plt.xticks(rotation=0)
plt.title("Test Scores by Model")
plt.legend(loc="lower right")
plt.show()
```

## Cell 62: Markdown

````markdown
*(Write your interpretation here.)*
````

## Cell 63: Markdown

````markdown
## 10. Hyperparameter Tuning (Q7)
````

## Cell 64: Markdown

````markdown
Two hyperparameters of KNN are tuned:
- `n_neighbors` (K): how many nearest patients vote.
- `weights`: `uniform` gives every neighbour an equal vote, `distance` gives closer neighbours a stronger vote.

Each pair is scored with 5-fold cross-validation on the training data, so the test set is not used for tuning.
````

## Cell 65: Code

```python
# K is the number of nearest patients that vote (n_neighbors in scikit-learn)
k_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 25]
weight_values = ["uniform", "distance"]

tuning_results = []

for weights in weight_values:
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, weights=weights)
        cv_scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)

        tuning_results.append({
            "n_neighbors": k,
            "weights": weights,
            "CV Accuracy": cv_scores.mean(),
        })

tuning_df = pd.DataFrame(tuning_results)

# Table: one row for each K, one column for each weights option
tuning_table = tuning_df.pivot(index="n_neighbors", columns="weights", values="CV Accuracy")
tuning_table.round(3)
```

## Cell 66: Code

```python
# Chart: K on the x-axis, one line for each weights option
plt.figure(figsize=(10, 5))

for weights in weight_values:
    part = tuning_df[tuning_df["weights"] == weights]
    plt.plot(part["n_neighbors"], part["CV Accuracy"], marker="o", label=f"weights = {weights}")

plt.xticks(k_values)
plt.xlabel("K (n_neighbors)")
plt.ylabel("CV Accuracy")
plt.title("KNN Tuning: CV Accuracy for Each K and weights")
plt.legend()
plt.show()
```

## Cell 67: Code

```python
# Best pair = the row with the highest CV accuracy
best_row = tuning_df.loc[tuning_df["CV Accuracy"].idxmax()]
best_k = int(best_row["n_neighbors"])
best_weights = best_row["weights"]

print("Best K (n_neighbors):", best_k)
print("Best weights        :", best_weights)
print("Best CV accuracy    :", round(best_row["CV Accuracy"], 3))
```

## Cell 68: Code

```python
# Train KNN again with the best pair and test it
best_knn = KNeighborsClassifier(n_neighbors=best_k, weights=best_weights)
best_knn.fit(X_train_scaled, y_train)

best_pred = best_knn.predict(X_test_scaled)
best_prob = best_knn.predict_proba(X_test_scaled)[:, 1]

# The KNN from Section 7 (n_neighbors=5, weights="uniform") for comparison
default_pred = predictions["KNN"]
default_prob = models["KNN"].predict_proba(X_test_scaled)[:, 1]

comparison = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
    "KNN before tuning": [
        accuracy_score(y_test, default_pred),
        precision_score(y_test, default_pred),
        recall_score(y_test, default_pred),
        f1_score(y_test, default_pred),
        roc_auc_score(y_test, default_prob),
    ],
    "KNN after tuning": [
        accuracy_score(y_test, best_pred),
        precision_score(y_test, best_pred),
        recall_score(y_test, best_pred),
        f1_score(y_test, best_pred),
        roc_auc_score(y_test, best_prob),
    ],
})

comparison.round(3)
```

## Cell 69: Markdown

````markdown
*(Write your interpretation here.)*
````

## Cell 70: Markdown

````markdown
## 11. Performance Across Groups (Q10)
````

## Cell 71: Markdown

````markdown
The tuned KNN is checked separately for younger and older patients. The number of patients in each group is shown too, because a score from a small group is less reliable.
````

## Cell 72: Code

```python
# One table with the age, the real answer and the prediction of the tuned KNN
test_results = pd.DataFrame({
    "age": age_test,
    "actual": y_test,
    "predicted": best_pred,
})

groups = {
    "Age below 50": test_results[test_results["age"] < 50],
    "Age 50 or above": test_results[test_results["age"] >= 50],
}

group_results = []

for group_name, group in groups.items():
    group_results.append({
        "Group": group_name,
        "Patients": len(group),
        "With kidney disease": group["actual"].sum(),
        "Accuracy": accuracy_score(group["actual"], group["predicted"]),
        "Recall": recall_score(group["actual"], group["predicted"], zero_division=0),
    })

pd.DataFrame(group_results).round(3)
```

## Cell 73: Markdown

````markdown
*(Write your interpretation here.)*
````

## Cell 74: Markdown

````markdown
## 12. Explainability: Feature Importance (Q10)
````

## Cell 75: Markdown

````markdown
KNN has no built-in feature importance, so permutation importance is used: the values of one feature are shuffled, and the drop in test accuracy shows how much the model depends on that feature.
````

## Cell 76: Code

```python
# Permutation importance of the tuned KNN on the test set
result = permutation_importance(best_knn, X_test_scaled, y_test,
                                n_repeats=30, random_state=42)

feature_importance = pd.Series(
    result.importances_mean, index=X.columns
).sort_values(ascending=False)

feature_importance.round(3)
```

## Cell 77: Code

```python
# Feature importance chart
plt.figure(figsize=(8, 7))
sns.barplot(x=feature_importance.values, y=feature_importance.index, color="seagreen")
plt.xlabel("Drop in accuracy when the feature is shuffled")
plt.title("Permutation Feature Importance (Tuned KNN)")
plt.show()
```

## Cell 78: Markdown

````markdown
*(Write your interpretation here.)*
````
