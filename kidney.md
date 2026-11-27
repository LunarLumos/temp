# Chronic Kidney Disease Screening - all notebook cells

This file has all 74 cells of the notebook in order: 40 text cells and 34 code cells.

How to use it on Kaggle:

1. Dataset: turn on the internet in Kaggle (**Settings** -> **Internet on**). The notebook then reads the dataset from the UCI website by itself. If you do not want to use the internet, upload `chronic_kidney_disease.csv` with **Add Input** -> **Upload**, and the notebook finds the file by itself.
2. For a **Text (Markdown)** cell: click **+ Markdown** in Kaggle, then copy the text inside the box and paste it.
3. For a **Code** cell: click **+ Code** in Kaggle, then copy the code inside the box and paste it.
4. Copy only what is inside the box. Do not copy the lines with the backticks.
5. Keep the same order as here (Cell 1, Cell 2, Cell 3, ...).

---

## Cell 1 - Text (Markdown)

````markdown
# Medical Screening Expert System for Chronic Kidney Disease

**Course:** CIS313 - Artificial Intelligence (Semester Assignment, Fall 2026)

**Topic:** No. 11 - Medical Screening Expert System for Chronic Kidney Disease

**Name:** A. Islam (Aadil)

**Student ID:** 242-16-025

**Section:** 20_A

**Dataset:** Chronic Kidney Disease dataset from the UCI Machine Learning Repository
https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease
````

---

## Cell 2 - Text (Markdown)

````markdown
# Task 2: Applying AI Algorithms and Machine Learning Techniques

## Q3. Explore and preprocess the dataset

### 3.1 About the dataset

The dataset has the blood and urine test results of 400 patients. One row is one patient.

**Input features (24):**

| Feature | Meaning | Type |
|---|---|---|
| age | Age in years | Number |
| bp | Blood pressure | Number |
| sg | Specific gravity of urine | Number |
| al | Albumin in urine (0 to 5) | Number |
| su | Sugar in urine (0 to 5) | Number |
| bgr | Blood glucose random | Number |
| bu | Blood urea | Number |
| sc | Serum creatinine | Number |
| sod | Sodium | Number |
| pot | Potassium | Number |
| hemo | Hemoglobin | Number |
| pcv | Packed cell volume | Number |
| wbcc | White blood cell count | Number |
| rbcc | Red blood cell count | Number |
| rbc | Red blood cells (normal or abnormal) | Text |
| pc | Pus cell (normal or abnormal) | Text |
| pcc | Pus cell clumps (present or notpresent) | Text |
| ba | Bacteria (present or notpresent) | Text |
| htn | Hypertension (yes or no) | Text |
| dm | Diabetes (yes or no) | Text |
| cad | Coronary artery disease (yes or no) | Text |
| appet | Appetite (good or poor) | Text |
| pe | Pedal edema, swelling of the feet (yes or no) | Text |
| ane | Anemia (yes or no) | Text |

**Target variable:** `class` (ckd = chronic kidney disease, notckd = no chronic kidney disease)

**Type of problem:** The target has only two values and the labels are given. So this is a supervised binary classification problem.
````

---

## Cell 3 - Text (Markdown)

````markdown
### 3.2 Import the libraries

First I import the libraries.
````

---

## Cell 4 - Code

```python
import os

# Data handling
import numpy as np
import pandas as pd

# Charts
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix)

sns.set_theme(style="whitegrid")
```

---

## Cell 5 - Text (Markdown)

````markdown
### 3.3 Load the dataset

Now I load the dataset and see the first 5 rows.

The code first looks for the file on my own computer. On Kaggle it looks inside the `/kaggle/input` folder. If the file is not found in both places, it reads the same file from the UCI website, so the notebook runs without any change.
````

---

## Cell 6 - Code

```python
# The same dataset on the UCI website
data_path = "https://archive.ics.uci.edu/static/public/336/data.csv"

# Path of the dataset on my own computer
if os.path.exists("data/chronic_kidney_disease.csv"):
    data_path = "data/chronic_kidney_disease.csv"

# On Kaggle: find the file inside the /kaggle/input folder
for folder, subfolders, files in os.walk("/kaggle/input"):
    if "chronic_kidney_disease.csv" in files:
        data_path = os.path.join(folder, "chronic_kidney_disease.csv")

print("Reading:", data_path)
df = pd.read_csv(data_path)

df.head()
```

---

## Cell 7 - Text (Markdown)

````markdown
One row is one patient. I can already see `NaN` in the first rows, so the dataset has missing values.

### 3.4 Dataset shape and data types
````

---

## Cell 8 - Code

```python
print("Shape:", df.shape)
df.info()
```

---

## Cell 9 - Text (Markdown)

````markdown
The dataset has 400 rows and 25 columns: 24 features and the `class` column. 14 columns are numbers and 11 columns are text. Every feature has fewer than 400 non-null values, so every feature has some missing values.

### 3.5 Summary statistics (feature summary table)
````

---

## Cell 10 - Code

```python
# Feature summary table
df.describe().round(2)
```

---

## Cell 11 - Text (Markdown)

````markdown
The average age is about 51 years and the patients are from 2 to 90 years old. The features have very different ranges. White blood cell count (`wbcc`) is in thousands but specific gravity (`sg`) is close to 1, so I need scaling later. Serum creatinine (`sc`) has a mean of 3.07 but a maximum of 76, so a few patients have very high values.
````

---

## Cell 12 - Text (Markdown)

````markdown
### 3.6 Clean the text values

Some text values have a hidden tab character (for example `"ckd\t"` and `"\tno"`). Because of this the same answer looks like two different values. `str.strip()` removes the extra spaces and tabs.

I keep the names of the text columns and the numeric columns in two lists, because I need them many times.
````

---

## Cell 13 - Code

```python
# Text columns and numeric columns
text_columns = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane", "class"]
numeric_columns = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot",
                   "hemo", "pcv", "wbcc", "rbcc"]

print("Before cleaning:", df["class"].unique())

for col in text_columns:
    df[col] = df[col].str.strip()

print("After cleaning :", df["class"].unique())
```

---

## Cell 14 - Text (Markdown)

````markdown
Before cleaning, the `class` column had three values because one of them had a hidden tab. Now it has only the two correct values: `ckd` and `notckd`.

### 3.7 Handle missing values and duplicates
````

---

## Cell 15 - Code

```python
# Check for missing values and duplicate rows
print("Missing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("Rows with no missing value at all:", len(df.dropna()))
```

---

## Cell 16 - Code

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

---

## Cell 17 - Text (Markdown)

````markdown
Every feature has missing values. Red blood cells (`rbc`, 152), red blood cell count (`rbcc`, 131) and white blood cell count (`wbcc`, 106) have the most. I think these tests were not done for every patient. There are 0 duplicate rows.

Only 158 of the 400 patients have no missing value at all. If I delete every row with a missing value, I lose 242 patients (about 60% of the data). So I fill the missing values:
- number columns: the median of the column
- text columns: the most common value of the column (mode)
````

---

## Cell 18 - Code

```python
# Number columns: fill with the median of the column
for col in numeric_columns:
    df[col] = df[col].fillna(df[col].median())

# Text columns: fill with the most common value of the column
for col in text_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

print("Missing values left:", df.isnull().sum().sum())
```

---

## Cell 19 - Text (Markdown)

````markdown
Now there are 0 missing values. I used the median for the number columns because it does not change much with very high values like serum creatinine 76.

### 3.8 Create the target

The `class` column is text. I change it to numbers: 1 = chronic kidney disease (ckd), 0 = no chronic kidney disease (notckd).
````

---

## Cell 20 - Code

```python
df["target"] = (df["class"] == "ckd").astype(int)

# The original column is no longer needed (keeping it would give the answer away)
df = df.drop(columns="class")

df["target"].value_counts()
```

---

## Cell 21 - Text (Markdown)

````markdown
250 patients (62.5%) have chronic kidney disease and 150 patients (37.5%) do not.
````

---

## Cell 22 - Text (Markdown)

````markdown
### 3.9 Chart 1: Class distribution
````

---

## Cell 23 - Code

```python
# Number of patients in each class
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="target")
plt.title("Class Distribution")
plt.xlabel("Target (0 = no kidney disease, 1 = kidney disease)")
plt.ylabel("Number of patients")
plt.show()
```

---

## Cell 24 - Text (Markdown)

````markdown
The two classes are not equal. There are more kidney disease patients (250) than patients without it (150). A model that says "kidney disease" for everyone would already get 62.5% accuracy. So I do not look only at accuracy. I also use precision, recall, F1-score and ROC-AUC.

### 3.10 Chart 2: Correlation heatmap

Correlation is a number between -1 and +1. Near +1 means two columns go up together. Near -1 means one goes up when the other goes down. Near 0 means no relation.
````

---

## Cell 25 - Code

```python
# Correlation between the number columns and the target
plt.figure(figsize=(13, 10))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Between Numeric Features and Target")
plt.show()
```

---

## Cell 26 - Code

```python
# Correlation of each numeric feature with the target, from highest to lowest
df.corr(numeric_only=True)["target"].drop("target").sort_values(ascending=False).round(2)
```

---

## Cell 27 - Text (Markdown)

````markdown
Hemoglobin (-0.73), packed cell volume (-0.67), specific gravity (-0.66) and red blood cell count (-0.57) have a strong negative correlation with the target. This means lower values go with kidney disease. Albumin (+0.53) has the strongest positive correlation. Potassium (+0.07) is almost not related to the target.

### 3.11 Chart 3: Hemoglobin and serum creatinine by target
````

---

## Cell 28 - Code

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

---

## Cell 29 - Code

```python
# Average hemoglobin and serum creatinine in each class
df.groupby("target")[["hemo", "sc"]].mean().round(2)
```

---

## Cell 30 - Text (Markdown)

````markdown
Kidney disease patients have lower hemoglobin (11.02 against 15.09) and much higher serum creatinine (4.27 against 0.88). The two boxes for hemoglobin almost do not overlap. So the two classes are easy to separate, and I expect high scores from the models.
````

---

## Cell 31 - Text (Markdown)

````markdown
### 3.12 Encode the text columns

A model can work only with numbers, so I change the text columns to 0/1 columns with one-hot encoding. Every text column has only two values (for example yes and no), so one 0/1 column is enough for each. That is why I use `drop_first=True`.
````

---

## Cell 32 - Code

```python
# All text columns except the target
categorical_columns = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]

df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True, dtype=int)

df_encoded.head()
```

---

## Cell 33 - Text (Markdown)

````markdown
Now all columns are numbers. For example `htn_yes` is 1 when the patient has hypertension and 0 when not.

I make `X` (the input features) and `y` (the target).
````

---

## Cell 34 - Code

```python
X = df_encoded.drop(columns="target")
y = df_encoded["target"]

print("X shape:", X.shape)
print("y shape:", y.shape)
```

---

## Cell 35 - Text (Markdown)

````markdown
### 3.13 Train and test split

The model learns from the training set. Then I check it on the testing set, which the model has never seen.
````

---

## Cell 36 - Code

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set:", X_train.shape)
print("Testing set :", X_test.shape)
```

---

## Cell 37 - Text (Markdown)

````markdown
80% of the data (320 patients) is for training and 20% (80 patients) is for testing. `random_state=42` gives the same split in every run. `stratify=y` keeps the same share of kidney disease patients in both sets.

I keep the real age of the test patients, because I need it in Q10.
````

---

## Cell 38 - Code

```python
# Keep the original (unscaled) age of the test patients for Q10
age_test = X_test["age"]
```

---

## Cell 39 - Text (Markdown)

````markdown
### 3.14 Scale the features

StandardScaler makes every feature have mean 0 and standard deviation 1, so all features are treated equally. I fit the scaler only on the training set and then use it on the test set. If the scaler learns from the test set also, information from the test set leaks into training (data leakage).
````

---

## Cell 40 - Code

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)   # learn from the training set
X_test_scaled = scaler.transform(X_test)         # use the same scaler on the test set
```

---

## Cell 41 - Text (Markdown)

````markdown
## Q4. Build and train at least two different models

I train five different models on the same training data:
- **Logistic Regression:** gives a weight to each feature and makes a probability. Above 0.5 means kidney disease.
- **KNN:** finds the 5 most similar patients and takes the majority class.
- **Decision Tree:** asks yes/no questions like a flowchart. These questions are the IF-THEN rules of the expert system.
- **Naive Bayes:** calculates the probability of each class and picks the larger one.
- **Random Forest:** makes 100 decision trees and takes the majority vote.
````

---

## Cell 42 - Code

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
    print(name, "is trained")
```

---

## Cell 43 - Text (Markdown)

````markdown
All five models are trained.

### 4.1 Rules of the expert system

A decision tree is a set of IF-THEN rules. I train the same tree on the unscaled values, so the rules show real medical values (class 1 = kidney disease, class 0 = no kidney disease).
````

---

## Cell 44 - Code

```python
rule_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
rule_tree.fit(X_train, y_train)

print(export_text(rule_tree, feature_names=list(X.columns)))
```

---

## Cell 45 - Text (Markdown)

````markdown
The tree made these rules by itself from the data:
- IF hemoglobin is 13.05 or lower AND red blood cell count is 5.10 or lower, THEN kidney disease.
- IF hemoglobin is above 13.05 AND specific gravity is 1.02 or lower, THEN kidney disease.
- IF hemoglobin is above 13.05 AND specific gravity is above 1.02 AND there is no albumin in urine, THEN no kidney disease.

A doctor can read and check these rules. This is the main strength of a rule-based expert system.
````

---

## Cell 46 - Text (Markdown)

````markdown
## Q5. Run the models and report the outputs

### 5.1 Training accuracy and test accuracy

If the training accuracy is much higher than the test accuracy, the model has memorised the training data. This is called overfitting.
````

---

## Cell 47 - Code

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

---

## Cell 48 - Text (Markdown)

````markdown
The training accuracy and the test accuracy are close for every model. The biggest gap is 1.6% (Decision Tree). So I do not see overfitting. All five models get at least 97.5% on the test set.

### 5.2 Confusion matrices
````

---

## Cell 49 - Code

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

---

## Cell 50 - Text (Markdown)

````markdown
Rows are the actual class and columns are the predicted class.
- **True Positive (TP):** the patient has kidney disease and the model says kidney disease.
- **True Negative (TN):** the patient is healthy and the model says no kidney disease.
- **False Positive (FP):** a healthy patient is wrongly flagged (false alarm).
- **False Negative (FN):** a kidney disease patient is missed. This is the most dangerous mistake because the patient goes home without treatment.

The test set has 80 patients (50 with kidney disease and 30 without). Random Forest makes no mistake. Logistic Regression misses 1 patient. KNN and Naive Bayes miss 2 patients each. Decision Tree misses 1 patient and gives 1 false alarm.

### 5.3 Classification reports
````

---

## Cell 51 - Code

```python
# Classification report of every model
for name in models:
    print("=====", name, "=====")
    print(classification_report(y_test, predictions[name], target_names=class_names))
```

---

## Cell 52 - Text (Markdown)

````markdown
The report gives precision, recall and F1-score for each class. For the CKD class the recall is between 0.96 and 1.00, so every model finds at least 48 of the 50 kidney disease patients.
````

---

## Cell 53 - Text (Markdown)

````markdown
# Task 3: Analyzing and Evaluating AI Model Performance

## Q6. Evaluate and compare the models with metrics

- **Accuracy** = (TP + TN) / all patients
- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **F1-score** balances precision and recall
- **ROC-AUC** tells how well the model separates the two classes (1.0 is perfect, 0.5 is guessing). It needs the predicted probability of class 1, so I use `predict_proba(...)[:, 1]`.
- **CV Accuracy** is the average accuracy of 5-fold cross-validation on the training data. It shows if a test score is stable or just lucky.
````

---

## Cell 54 - Code

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
        "CV Accuracy": cross_val_score(model, X_train_scaled, y_train, cv=5).mean(),
    })

results_df = pd.DataFrame(results).sort_values(by=["Accuracy", "F1 Score"], ascending=False)
results_df.round(3).reset_index(drop=True)
```

---

## Cell 55 - Code

```python
# Model comparison chart
results_df.plot.bar(x="Model", y=["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
                    figsize=(12, 5))
plt.ylim(0.9, 1.01)
plt.ylabel("Score (axis starts at 0.9)")
plt.xlabel("")
plt.xticks(rotation=0)
plt.title("Test Scores by Model")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5)
plt.show()
```

---

## Cell 56 - Text (Markdown)

````markdown
All five models are very close. The test accuracy is between 0.975 and 1.000, which is a difference of only 2 patients out of 80.

**Random Forest is the best model.** It has 1.00 in all five test metrics, and it is also first in cross-validation (0.988, the same as Logistic Regression). So its good result is not only luck in one split.

For this problem recall is the most important metric, because a missed kidney disease patient is more dangerous than a false alarm. Random Forest is the only model with recall 1.00. KNN and Naive Bayes have the lowest recall (0.96).
````

---

## Cell 57 - Text (Markdown)

````markdown
## Q7. Tune two hyperparameters

Random Forest already has a test score of 1.000, so tuning cannot improve it. I tune KNN, because it has the lowest recall and its main hyperparameter is K. I want to see if tuning can bring it close to the best model.

I tune two hyperparameters:
- `n_neighbors` (K): how many nearest patients vote.
- `weights`: `uniform` gives every neighbour an equal vote, `distance` gives closer neighbours a stronger vote.

I check each pair with 5-fold cross-validation on the training data, so the test set is not used for tuning.
````

---

## Cell 58 - Code

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

---

## Cell 59 - Code

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

---

## Cell 60 - Text (Markdown)

````markdown
The accuracy is highest at K = 1 (0.975) and goes down when K gets bigger. At K = 25 it is only 0.903 with uniform weights. The two weights options give the same result up to K = 13. After that, `distance` is a little better.

Usually a very small K overfits, because one noisy neighbour decides the answer. Here it is different. The two classes are well separated (Chart 3 in Q3), so the nearest patient is almost always from the correct class. When K is large, many neighbours vote. Then the healthy patients can outvote the few kidney disease patients who have mild values, and the model becomes too simple (underfitting).

Now I take the best pair.
````

---

## Cell 61 - Code

```python
# Best pair = the row with the highest CV accuracy
best_row = tuning_df.loc[tuning_df["CV Accuracy"].idxmax()]
best_k = int(best_row["n_neighbors"])
best_weights = best_row["weights"]

print("Best K (n_neighbors):", best_k)
print("Best weights        :", best_weights)
print("Best CV accuracy    :", round(best_row["CV Accuracy"], 3))
```

---

## Cell 62 - Text (Markdown)

````markdown
The best pair is K = 1. With K = 1 both weights options give the same result, because only one neighbour votes. I train KNN again with the best pair and test it.
````

---

## Cell 63 - Code

```python
# Train KNN again with the best pair and test it
best_knn = KNeighborsClassifier(n_neighbors=best_k, weights=best_weights)
best_knn.fit(X_train_scaled, y_train)

best_pred = best_knn.predict(X_test_scaled)
best_prob = best_knn.predict_proba(X_test_scaled)[:, 1]

# The KNN from Q4 (n_neighbors=5, weights="uniform") for comparison
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

---

## Cell 64 - Text (Markdown)

````markdown
After tuning, the test accuracy went up from 0.975 to 0.988 and the recall went up from 0.96 to 0.98. So the tuned KNN misses 1 kidney disease patient, not 2. The precision is still 1.00, so there is no false alarm.

The tuned KNN (0.988) is still below Random Forest (1.000).
````

---

## Cell 65 - Text (Markdown)

````markdown
## Q8. Strengths, limitations and the most suitable model

I compare the five models with the results from Q5, Q6 and Q7.

| Model | Strength | Limitation |
|---|---|---|
| Random Forest | Best score in every metric, no missed patient | 100 trees are hard to read |
| Logistic Regression | Simple and gives a probability | Missed 1 kidney disease patient |
| KNN | Easy idea (similar patients), no false alarm | Lowest recall, and it must keep all patient records |
| Decision Tree | Its rules are easy to read | 1 missed patient and 1 false alarm |
| Naive Bayes | Very fast | Lowest recall, and it treats the features as independent |

The main trade-off is accuracy against interpretability. Random Forest is the most accurate but the hardest to explain. Decision Tree is the easiest to explain but makes 2 mistakes. All models train in less than a second on this small dataset, so training time does not decide the choice.

I take the model with the best scores as my final model.
````

---

## Cell 66 - Code

```python
# The final model is the model with the best scores in Q6
final_name = "Random Forest"
final_model = models[final_name]
final_pred = predictions[final_name]

# Scores of the final model
results_df[results_df["Model"] == final_name].round(3)
```

---

## Cell 67 - Text (Markdown)

````markdown
**I select Random Forest.** It is first in every test metric and first in cross-validation. It is also the only model that did not miss any kidney disease patient, and recall is the most important metric here.

Its limitation is that nobody can read 100 trees. So I explain it in two ways: the decision tree rules in Q4 and the feature importance in Q10.
````

---

## Cell 68 - Text (Markdown)

````markdown
# Task 4: Designing a Responsible Intelligent System

## Q10. Performance across groups and explainability

### 10.1 Performance across two groups

I check the final model (Random Forest) separately for younger and older patients. I also show the number of patients in each group, because a score from a small group is less reliable.
````

---

## Cell 69 - Code

```python
# One table with the age, the real answer and the prediction of the final model
test_results = pd.DataFrame({
    "age": age_test,
    "actual": y_test,
    "predicted": final_pred,
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

---

## Cell 70 - Text (Markdown)

````markdown
Random Forest is correct for every test patient in both groups, so I see no difference between younger and older patients.

But I am careful with this result. The younger group has only 29 patients and only 10 of them have kidney disease. One mistake would change its recall from 1.00 to 0.90. I need a bigger test set to be sure that the model is equally good for both groups.

The dataset has no sex column, so I cannot check the model for men and women.
````

---

## Cell 71 - Text (Markdown)

````markdown
### 10.2 Explainability: feature importance

A Random Forest has a built-in feature importance. It shows how much each feature helps the trees to separate the two classes. The values add up to 1.
````

---

## Cell 72 - Code

```python
# Feature importance of the final model (Random Forest)
feature_importance = pd.Series(
    final_model.feature_importances_, index=X.columns
).sort_values(ascending=False)

feature_importance.round(3)
```

---

## Cell 73 - Code

```python
# Feature importance chart
plt.figure(figsize=(8, 7))
sns.barplot(x=feature_importance.values, y=feature_importance.index, color="seagreen")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance (Random Forest)")
plt.show()
```

---

## Cell 74 - Text (Markdown)

````markdown
The five most important features are hemoglobin (0.214), packed cell volume (0.174), red blood cell count (0.126), serum creatinine (0.109) and specific gravity (0.108). Together they have about 73% of the importance.

These features make sense medically. Serum creatinine and specific gravity show how well the kidneys work, and the three blood values show the anemia that kidney disease causes. They are also the features with the strongest correlation in Chart 2 of Q3, and the decision tree uses hemoglobin, red blood cell count and specific gravity in its rules. So three different checks agree.

Age has an importance of only 0.009, so the model does not decide from the age of the patient.
````
