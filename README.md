# Heart Disease Prediction - all notebook cells

This file has all 97 cells of the notebook in order: 50 text cells and 47 code cells.

How to use it on Kaggle:

1. Add the dataset first: **Add Input** -> search `heart failure prediction` -> add the dataset by fedesoriano.
2. For a **Text (Markdown)** cell: click **+ Markdown** in Kaggle, then copy the text inside the box and paste it.
3. For a **Code** cell: click **+ Code** in Kaggle, then copy the code inside the box and paste it.
4. Copy only what is inside the box. Do not copy the lines with the backticks.
5. Keep the same order as here (Cell 1, Cell 2, Cell 3, ...).

---

## Cell 1 - Text (Markdown)

```markdown
# Heart Disease Prediction Using Machine Learning

**Course:** BI334L - Data Analysis and Business Modeling Lab (Lab Project, Fall 2026)

**Name:** _(write your name)_

**Student ID:** _(write your ID)_

**Dataset:** Heart Failure Prediction Dataset from Kaggle
https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
```

---

## Cell 2 - Text (Markdown)

```markdown
# 1. Problem Understanding

**Background:** Heart disease is the number one cause of death in the world. About 17.9 million people die from it every year (WHO). Many lives can be saved if it is found early. But a full check-up needs costly tests and specialist doctors, and a hospital has limited doctors, time and money.

**Objectives:**
- Find which factors are related to heart disease.
- Clean and prepare the data.
- Build at least three machine learning models.
- Compare the models and find the best one.

**Real-world application:** A hospital can use the model as a screening tool. High-risk patients can be sent to a heart doctor first. The model only helps the doctor. It does not replace the doctor.

**Input features (11):**

| Feature | Meaning |
|---|---|
| Age | Age in years |
| Sex | M = male, F = female |
| ChestPainType | TA, ATA, NAP or ASY (ASY means no chest pain) |
| RestingBP | Resting blood pressure |
| Cholesterol | Serum cholesterol |
| FastingBS | 1 if fasting blood sugar is above 120, else 0 |
| RestingECG | Normal, ST or LVH |
| MaxHR | Maximum heart rate in exercise |
| ExerciseAngina | Chest pain in exercise (Y or N) |
| Oldpeak | ST depression in exercise |
| ST_Slope | Up, Flat or Down |

**Target variable:** `HeartDisease` (1 = heart disease, 0 = normal)

**Type of problem:** The target has only two values and the labels are given. So this is a supervised binary classification problem.
```

---

## Cell 3 - Text (Markdown)

```markdown
# 2. Dataset Exploration

First I import the libraries.
```

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

# Statistics
from scipy import stats

# Machine learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, classification_report)

sns.set_style("whitegrid")
```

---

## Cell 5 - Text (Markdown)

```markdown
Now I load the dataset and see the first 5 rows.

On Kaggle the dataset is inside the `/kaggle/input` folder. The code below looks for `heart.csv` in that folder, so the notebook runs on Kaggle without any change.
```

---

## Cell 6 - Code

```python
# Path of the dataset on my own computer
data_path = "dataset/heart.csv"

# On Kaggle: find heart.csv inside the /kaggle/input folder
for folder, subfolders, files in os.walk("/kaggle/input"):
    if "heart.csv" in files:
        data_path = os.path.join(folder, "heart.csv")

print("Reading:", data_path)
df = pd.read_csv(data_path)

df.head()
```

---

## Cell 7 - Text (Markdown)

```markdown
### Dataset shape
```

---

## Cell 8 - Code

```python
rows, columns = df.shape

print("Rows   :", rows)
print("Columns:", columns)
```

---

## Cell 9 - Text (Markdown)

```markdown
The dataset has 918 rows and 12 columns. One row is one patient.

### Data types
```

---

## Cell 10 - Code

```python
df.info()
```

---

## Cell 11 - Text (Markdown)

```markdown
7 columns are numbers and 5 columns are text. There are no empty cells.

I keep the names of the numeric columns and the text columns in two lists, because I need them many times.
```

---

## Cell 12 - Code

```python
# Numeric columns and text (categorical) columns
num_cols = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
cat_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
```

---

## Cell 13 - Text (Markdown)

```markdown
### Summary statistics
```

---

## Cell 14 - Code

```python
df.describe().round(2)
```

---

## Cell 15 - Text (Markdown)

```markdown
The average age is about 53.5 years. The minimum of `RestingBP` and `Cholesterol` is 0. This is not possible for a living person, so I fix it in Section 3.

Now I see the text columns.
```

---

## Cell 16 - Code

```python
# Number of patients in each category
for col in cat_cols:
    print(df[col].value_counts())
    print()
```

---

## Cell 17 - Text (Markdown)

```markdown
Most patients are male. The most common chest pain type is ASY.

### Class distribution
```

---

## Cell 18 - Code

```python
# Number and percentage of patients in each class
class_counts = df["HeartDisease"].value_counts()
class_percent = (class_counts / len(df) * 100).round(1)

pd.DataFrame({"Patients": class_counts, "Percent": class_percent})
```

---

## Cell 19 - Code

```python
sns.countplot(data=df, x="HeartDisease")
plt.title("Class distribution of the target")
plt.xlabel("HeartDisease (0 = Normal, 1 = Heart disease)")
plt.ylabel("Number of patients")
plt.show()
```

---

## Cell 20 - Text (Markdown)

```markdown
508 patients (55.3%) have heart disease and 410 patients (44.7%) are normal. The two classes are close, so the dataset is fairly balanced.
```

---

## Cell 21 - Text (Markdown)

```markdown
# 3. Data Cleaning and Preprocessing

## 3.1 Handle missing values
```

---

## Cell 22 - Code

```python
# Empty cells in each column
df.isnull().sum()
```

---

## Cell 23 - Text (Markdown)

```markdown
There are no empty cells. But I saw 0 in `Cholesterol` and `RestingBP` before. These are hidden missing values. I count them.
```

---

## Cell 24 - Code

```python
# A value of 0 is not possible in these two columns
print("Rows with Cholesterol = 0:", (df["Cholesterol"] == 0).sum())
print("Rows with RestingBP = 0  :", (df["RestingBP"] == 0).sum())
```

---

## Cell 25 - Text (Markdown)

```markdown
172 rows have cholesterol = 0 and 1 row has blood pressure = 0. First I change these zeros to NaN.
```

---

## Cell 26 - Code

```python
# Change the zeros to NaN (missing)
df["Cholesterol"] = df["Cholesterol"].replace(0, np.nan)
df["RestingBP"] = df["RestingBP"].replace(0, np.nan)

df.isnull().sum()
```

---

## Cell 27 - Text (Markdown)

```markdown
Now I fill them with the median of the column.
```

---

## Cell 28 - Code

```python
# Fill the missing values with the median of the column
chol_median = df["Cholesterol"].median()
bp_median = df["RestingBP"].median()

df["Cholesterol"] = df["Cholesterol"].fillna(chol_median)
df["RestingBP"] = df["RestingBP"].fillna(bp_median)

print("Cholesterol median:", chol_median)
print("RestingBP median  :", bp_median)
print("Missing values now:", df.isnull().sum().sum())
```

---

## Cell 29 - Text (Markdown)

```markdown
I did not delete the 172 rows because that would remove about 19% of the data. I used the median because it does not change much with very high values.

## 3.2 Remove duplicates
```

---

## Cell 30 - Code

```python
print("Duplicate rows:", df.duplicated().sum())

df = df.drop_duplicates()
print("Shape after removing duplicates:", df.shape)
```

---

## Cell 31 - Text (Markdown)

```markdown
I checked duplicates because duplicate rows make the model count the same patient two times. There are 0 duplicate rows.

## 3.3 Detect outliers

An outlier is a value that is very far from the other values. First I look at the boxplots.
```

---

## Cell 32 - Code

```python
# One boxplot for each numeric column
fig, axes = plt.subplots(1, 5, figsize=(18, 4))

for ax, col in zip(axes, num_cols):
    sns.boxplot(y=df[col], ax=ax)
    ax.set_title(col)

plt.tight_layout()
plt.show()
```

---

## Cell 33 - Text (Markdown)

```markdown
Now I count the outliers with the IQR rule. A value is an outlier if it is more than 1.5 times the IQR below Q1 or above Q3.
```

---

## Cell 34 - Code

```python
# IQR rule
for col in num_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    is_outlier = (df[col] < lower) | (df[col] > upper)

    print(f"{col}: range {lower:.2f} to {upper:.2f}, outliers = {is_outlier.sum()}")
```

---

## Cell 35 - Text (Markdown)

```markdown
I also count them with the z-score. A value is an outlier if it is more than 3 standard deviations away from the mean.
```

---

## Cell 36 - Code

```python
# Z-score rule
for col in num_cols:
    z_scores = np.abs(stats.zscore(df[col]))
    is_outlier = z_scores > 3

    print(f"{col}: outliers = {is_outlier.sum()}")
```

---

## Cell 37 - Text (Markdown)

```markdown
Cholesterol, RestingBP and Oldpeak have the most outliers. Age has none.

I kept the outliers because they are real medical values. A heart disease model should learn from such high values. The wrong values (the zeros) are already fixed.

## 3.4 Encode categorical variables

A model can work only with numbers, so I change the 5 text columns to numbers.
- `Sex` and `ExerciseAngina` have two values, so I change them to 0 and 1.
- `ChestPainType`, `RestingECG` and `ST_Slope` have more than two values, so I use one-hot encoding.

I save the result in `df_encoded`. I keep `df` as it is because the text labels are easier to read in the charts.
```

---

## Cell 38 - Code

```python
df_encoded = df.copy()

# Columns with two values -> 0 and 1
df_encoded["Sex"] = df_encoded["Sex"].map({"M": 1, "F": 0})
df_encoded["ExerciseAngina"] = df_encoded["ExerciseAngina"].map({"Y": 1, "N": 0})

# Columns with more than two values -> one-hot encoding
one_hot_cols = ["ChestPainType", "RestingECG", "ST_Slope"]
df_encoded = pd.get_dummies(df_encoded, columns=one_hot_cols, dtype=int)

print("Shape after encoding:", df_encoded.shape)
df_encoded.head()
```

---

## Cell 39 - Text (Markdown)

```markdown
I used one-hot encoding because simple numbers like 0, 1, 2, 3 can confuse the model. The model would think that 3 is bigger than 0, which has no meaning here. Now the data has 19 columns and all are numbers.

## 3.5 Scale the features

The features have very different ranges. Cholesterol goes up to 603 but Oldpeak goes up to only 6.2. So I use StandardScaler. It makes every feature have mean 0 and standard deviation 1.
```

---

## Cell 40 - Code

```python
# Small demo to show what StandardScaler does
scaled_values = StandardScaler().fit_transform(df[num_cols])
scaled_df = pd.DataFrame(scaled_values, columns=num_cols)

print("Before scaling:")
print(df[num_cols].agg(["mean", "std"]).round(2))
print()
print("After scaling:")
print(scaled_df.agg(["mean", "std"]).round(2))
```

---

## Cell 41 - Text (Markdown)

```markdown
After scaling every column has mean 0 and standard deviation 1, so all features are treated equally.

This cell only shows what scaling does. For the models I do the scaling in Section 6 after the train/test split, because the scaler should learn only from the training data.
```

---

## Cell 42 - Text (Markdown)

```markdown
# 4. Exploratory Data Analysis (EDA)

## 4.1 Histograms
```

---

## Cell 43 - Code

```python
df[num_cols].hist(bins=20, figsize=(14, 8), edgecolor="black")
plt.suptitle("Distribution of numeric features")
plt.tight_layout()
plt.show()
```

---

## Cell 44 - Text (Markdown)

```markdown
Age and MaxHR look close to a bell shape. Oldpeak has most values near 0. Cholesterol has one very tall bar near 237. This bar comes from the 172 missing values that I filled with the median.

## 4.2 Count plots
```

---

## Cell 45 - Code

```python
# One count plot for each categorical column, split by the target
plot_cols = cat_cols + ["FastingBS"]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))

for ax, col in zip(axes.flatten(), plot_cols):
    sns.countplot(data=df, x=col, hue="HeartDisease", ax=ax)
    ax.set_title(col + " vs HeartDisease")

plt.tight_layout()
plt.show()
```

---

## Cell 46 - Text (Markdown)

```markdown
I also calculate the percentage of heart disease in each category.
```

---

## Cell 47 - Code

```python
# Percentage of patients with heart disease in each category
for col in plot_cols:
    rate = df.groupby(col)["HeartDisease"].mean() * 100
    print(rate.round(1))
    print()
```

---

## Cell 48 - Text (Markdown)

```markdown
What I found:
- Men have a much higher rate than women (63.2% against 25.9%).
- ASY chest pain has the highest rate (79.0%). So no pain does not mean the patient is safe.
- Chest pain in exercise is a strong sign (85.2%).
- Flat and Down ST slopes have a high rate. Up slope is mostly normal.
- RestingECG groups are close to each other, so this feature is weak.

## 4.3 Boxplots by target
```

---

## Cell 49 - Code

```python
# One boxplot for each numeric column, split by the target
fig, axes = plt.subplots(1, 5, figsize=(18, 4))

for ax, col in zip(axes, num_cols):
    sns.boxplot(data=df, x="HeartDisease", y=col, ax=ax)
    ax.set_title(col)

plt.tight_layout()
plt.show()
```

---

## Cell 50 - Code

```python
# Average of each numeric column in the two classes
df.groupby("HeartDisease")[num_cols].mean().round(2)
```

---

## Cell 51 - Text (Markdown)

```markdown
Heart disease patients are older (55.9 against 50.6 years). Their maximum heart rate is lower (127.7 against 148.2) and their Oldpeak is higher (1.27 against 0.41). RestingBP and Cholesterol are almost the same in both groups.

## 4.4 Scatter plot
```

---

## Cell 52 - Code

```python
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="Age", y="MaxHR", hue="HeartDisease", alpha=0.7)
plt.title("Age vs Maximum Heart Rate")
plt.show()
```

---

## Cell 53 - Text (Markdown)

```markdown
When age goes up, the maximum heart rate goes down. Heart disease patients are mostly in the lower right side. But the two groups overlap a lot, so the model needs the other features also.

## 4.5 Correlation matrix (heatmap)

Correlation is a number between -1 and +1. Near +1 means two columns go up together. Near -1 means one goes up when the other goes down. Near 0 means no relation.
```

---

## Cell 54 - Code

```python
corr_matrix = df[num_cols + ["FastingBS", "HeartDisease"]].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation matrix")
plt.show()
```

---

## Cell 55 - Text (Markdown)

```markdown
Oldpeak (+0.40) and MaxHR (-0.40) have the strongest correlation with the target. Age (+0.28) and FastingBS (+0.27) are medium. RestingBP and Cholesterol are very weak. No two input features are highly correlated with each other.

**Important findings of the EDA:** ST_Slope, ChestPainType, ExerciseAngina, Oldpeak and MaxHR are the features most related to heart disease. RestingECG, RestingBP and Cholesterol are the weakest.
```

---

## Cell 56 - Text (Markdown)

```markdown
# 5. Feature Engineering

## 5.1 Feature creation

I make two new features.
- `HR_Ratio`: the expected maximum heart rate of a person is about 220 minus age. I divide MaxHR by this value. It shows how much of the expected heart rate the patient reached.
- `HighCholesterol`: 1 if cholesterol is above 240, else 0. Doctors call cholesterol above 240 high.
```

---

## Cell 57 - Code

```python
# New feature 1: share of the expected maximum heart rate (220 - age) that was reached
df_encoded["HR_Ratio"] = df_encoded["MaxHR"] / (220 - df_encoded["Age"])

# New feature 2: 1 if cholesterol is above 240, else 0
df_encoded["HighCholesterol"] = (df_encoded["Cholesterol"] > 240).astype(int)

df_encoded[["Age", "MaxHR", "HR_Ratio", "Cholesterol", "HighCholesterol"]].head()
```

---

## Cell 58 - Code

```python
# Average of the new features in the two classes
df_encoded.groupby("HeartDisease")[["HR_Ratio", "HighCholesterol"]].mean().round(3)
```

---

## Cell 59 - Text (Markdown)

```markdown
I made these features because a new feature can show information that the raw columns do not show directly. Normal patients reach about 87% of their expected heart rate but heart disease patients reach only about 78%. So `HR_Ratio` looks useful. `HighCholesterol` is almost the same in both groups, so it does not look useful.

## 5.2 Feature selection

All features do not help the model. I calculate the correlation of every feature with the target. Then I keep only the features with a correlation of 0.1 or more (I ignore the plus or minus sign).
```

---

## Cell 60 - Code

```python
# Correlation of every feature with the target
correlation = df_encoded.corr()["HeartDisease"].drop("HeartDisease")
correlation = correlation.sort_values()

correlation.plot(kind="barh", figsize=(8, 7))
plt.title("Correlation of each feature with HeartDisease")
plt.xlabel("Correlation")
plt.show()
```

---

## Cell 61 - Code

```python
# Keep the features with a correlation of 0.1 or more (sign ignored)
selected_features = correlation[correlation.abs() >= 0.1].index.tolist()
dropped_features = correlation[correlation.abs() < 0.1].index.tolist()

print(f"Selected features ({len(selected_features)}):")
print(selected_features)
print()
print(f"Dropped features ({len(dropped_features)}):")
print(dropped_features)
```

---

## Cell 62 - Text (Markdown)

```markdown
15 features are selected and 5 are dropped. I dropped the weak features because fewer and stronger features make the model simpler. My new feature `HR_Ratio` is kept but `HighCholesterol` is dropped.

Now I make `X` (the input features) and `y` (the target).
```

---

## Cell 63 - Code

```python
X = df_encoded[selected_features]
y = df_encoded["HeartDisease"]

print("X shape:", X.shape)
print("y shape:", y.shape)
```

---

## Cell 64 - Text (Markdown)

```markdown
## 5.3 Dimensionality reduction with PCA (optional)

PCA joins many features into a few new features. I reduce the 15 features to 2 components only to draw the patients on a 2D chart.
```

---

## Cell 65 - Code

```python
# PCA needs scaled data
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

explained = pca.explained_variance_ratio_ * 100
print(f"Component 1 keeps {explained[0]:.1f}% of the information")
print(f"Component 2 keeps {explained[1]:.1f}% of the information")
print(f"Total: {explained.sum():.1f}%")
```

---

## Cell 66 - Code

```python
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, alpha=0.7)
plt.title("Patients shown on 2 principal components")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
```

---

## Cell 67 - Text (Markdown)

```markdown
The two components keep only about 37% of the information. Still, normal patients are mostly on one side and heart disease patients are mostly on the other side.

I did not use the PCA components for training because about 63% of the information is lost. I train the models on all 15 selected features.
```

---

## Cell 68 - Text (Markdown)

```markdown
# 6. Machine Learning Model Development

## 6.1 Train and test split

The model learns from the training set. Then I check it on the testing set, which the model has never seen.
```

---

## Cell 69 - Code

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("Training set:", X_train.shape)
print("Testing set :", X_test.shape)
```

---

## Cell 70 - Text (Markdown)

```markdown
80% of the data (734 patients) is for training and 20% (184 patients) is for testing. `random_state=42` gives the same split in every run. `stratify=y` keeps the same share of heart disease patients in both sets.

## 6.2 Scale the features

I fit the scaler only on the training set and then use it on the test set.
```

---

## Cell 71 - Code

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)   # learn from the training set
X_test_scaled = scaler.transform(X_test)         # use the same scaler on the test set
```

---

## Cell 72 - Text (Markdown)

```markdown
## 6.3 Build the models

I train four models:
- **Logistic Regression:** gives a weight to each feature and makes a probability. Above 0.5 means heart disease.
- **KNN:** finds the 5 most similar patients and takes the majority class.
- **Decision Tree:** asks yes/no questions like a flowchart.
- **Random Forest:** makes 100 decision trees and takes the majority vote.
```

---

## Cell 73 - Code

```python
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    print(name, "is trained")
```

---

## Cell 74 - Text (Markdown)

```markdown
## 6.4 Compare the models

First I compare the training accuracy and the testing accuracy. If the training accuracy is much higher, the model has memorised the training data. This is called overfitting.
```

---

## Cell 75 - Code

```python
# Training accuracy and testing accuracy of each model
comparison = []

for name, model in models.items():
    comparison.append({
        "Model": name,
        "Training accuracy": model.score(X_train_scaled, y_train),
        "Testing accuracy": model.score(X_test_scaled, y_test),
    })

comparison_df = pd.DataFrame(comparison).set_index("Model").round(3)
comparison_df
```

---

## Cell 76 - Text (Markdown)

```markdown
Logistic Regression has the highest testing accuracy (89.1%). Random Forest has 100% training accuracy but only 84.8% testing accuracy, so it is overfitting. The other three models are not overfitting.
```

---

## Cell 77 - Text (Markdown)

```markdown
# 7. Model Evaluation

## 7.1 Evaluation metrics

- **TP:** sick patient predicted as sick
- **TN:** healthy patient predicted as healthy
- **FP:** healthy patient predicted as sick
- **FN:** sick patient predicted as healthy

The metrics:
- **Accuracy** = (TP + TN) / all patients
- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **F1-score** balances precision and recall
- **ROC-AUC** tells how well the model separates the two classes (1.0 is perfect, 0.5 is guessing)
```

---

## Cell 78 - Code

```python
# Predictions of every model on the test set (used in the cells below)
predictions = {}      # predicted class (0 or 1)
probabilities = {}    # predicted probability of class 1

for name, model in models.items():
    predictions[name] = model.predict(X_test_scaled)
    probabilities[name] = model.predict_proba(X_test_scaled)[:, 1]
```

---

## Cell 79 - Code

```python
# The five metrics of each model
results = []

for name in models:
    y_pred = predictions[name]
    y_prob = probabilities[name]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    })

results_df = pd.DataFrame(results).set_index("Model").round(3)
results_df
```

---

## Cell 80 - Code

```python
results_df.plot(kind="bar", figsize=(11, 5))
plt.title("Model comparison on the test set")
plt.ylabel("Score (axis starts at 0.7)")
plt.ylim(0.7, 1.0)
plt.xticks(rotation=0)
plt.legend(loc="upper right", ncol=5)
plt.show()
```

---

## Cell 81 - Text (Markdown)

```markdown
## 7.2 Confusion matrices
```

---

## Cell 82 - Code

```python
# One confusion matrix for each model
fig, axes = plt.subplots(1, 4, figsize=(18, 4))

for ax, name in zip(axes, models):
    cm = confusion_matrix(y_test, predictions[name])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.show()
```

---

## Cell 83 - Text (Markdown)

```markdown
Rows are the actual class and columns are the predicted class.

Logistic Regression found 93 of the 102 heart disease patients and missed 9. It gave 11 false alarms. It has the fewest missed patients and the fewest false alarms. A missed patient (FN) is the most dangerous mistake because the patient goes home without treatment.
```

---

## Cell 84 - Text (Markdown)

```markdown
## 7.3 ROC curves
```

---

## Cell 85 - Code

```python
plt.figure(figsize=(7, 6))

for name in models:
    fpr, tpr, thresholds = roc_curve(y_test, probabilities[name])
    auc = roc_auc_score(y_test, probabilities[name])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.title("ROC curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.legend()
plt.show()
```

---

## Cell 86 - Text (Markdown)

```markdown
A curve closer to the top-left corner is better. All four models are far above the dashed line.

## 7.4 Best-performing model
```

---

## Cell 87 - Code

```python
# The model with the highest F1-score
best_model_name = results_df["F1-score"].idxmax()

print("Best model based on F1-score:", best_model_name)
print()
print(results_df.loc[best_model_name])
```

---

## Cell 88 - Code

```python
print("Classification report for", best_model_name)
print()
print(classification_report(y_test, predictions[best_model_name],
                            target_names=["Normal", "Heart disease"]))
```

---

## Cell 89 - Text (Markdown)

```markdown
**Logistic Regression is the best model.** I used the F1-score to choose because it balances precision and recall. Logistic Regression is also the highest in the other four metrics.

It predicted 164 of the 184 test patients correctly and found 93 of the 102 heart disease patients.

I think the simple model won because the dataset is small (918 patients). A complex model like Random Forest overfits more easily on small data.
```

---

## Cell 90 - Text (Markdown)

```markdown
# 8. Bonus Challenges

## 8.1 Cross-validation

The result above comes from only one split, so it can be a little lucky. In 5-fold cross-validation the training data is cut into 5 parts. The model trains on 4 parts and is checked on the last part. This is done 5 times and the average is taken.
```

---

## Cell 91 - Code

```python
# 5-fold cross-validation accuracy of each model
cv_results = []

for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="accuracy")

    cv_results.append({
        "Model": name,
        "CV mean accuracy": scores.mean(),
        "CV std": scores.std(),
    })

cv_df = pd.DataFrame(cv_results).set_index("Model").round(3)
cv_df
```

---

## Cell 92 - Text (Markdown)

```markdown
Random Forest, KNN and Logistic Regression are within about 1% of each other. So cross-validation cannot say that one of them is clearly better. Decision Tree is the weakest.

The cross-validation accuracy of Logistic Regression (84.6%) is lower than its test accuracy (89.1%). So for new patients I expect about 85% to 89% accuracy.
```

---

## Cell 93 - Text (Markdown)

```markdown
## 8.2 Hyperparameter tuning

Random Forest was overfitting because its trees have no depth limit by default. So I tune it with GridSearchCV. It tries all combinations of the values below and checks each one with 5-fold cross-validation.

- `n_estimators`: number of trees (100 or 200)
- `max_depth`: depth of each tree (3, 4, 5, 6 or 8)
```

---

## Cell 94 - Code

```python
# Values to try
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 4, 5, 6, 8],
}

grid = GridSearchCV(RandomForestClassifier(random_state=42),
                    param_grid, cv=5, scoring="accuracy")
grid.fit(X_train_scaled, y_train)

print("Best parameters :", grid.best_params_)
print("Best CV accuracy:", round(grid.best_score_, 3))
```

---

## Cell 95 - Code

```python
# Random Forest before and after tuning
forests = {
    "Random Forest (default)": models["Random Forest"],
    "Random Forest (tuned)": grid.best_estimator_,
}

tuning = []

for name, model in forests.items():
    tuning.append({
        "Model": name,
        "Training accuracy": model.score(X_train_scaled, y_train),
        "Testing accuracy": model.score(X_test_scaled, y_test),
    })

tuning_df = pd.DataFrame(tuning).set_index("Model").round(3)
tuning_df
```

---

## Cell 96 - Text (Markdown)

```markdown
The best combination is `max_depth = 8` and `n_estimators = 200`. The training accuracy came down from 100% to 96.6% and the testing accuracy went up from 84.8% to 85.9%. So the overfitting went down a little. But the tuned Random Forest is still below Logistic Regression.
```

---

## Cell 97 - Text (Markdown)

```markdown
# 9. Discussion and Conclusion

**Summary of findings:**
- The exercise test results (ST slope, exercise angina, Oldpeak, maximum heart rate) and the chest pain type are the strongest signs of heart disease.
- Logistic Regression is the best model with 89.1% accuracy, 91.2% recall, 0.903 F1-score and 0.933 ROC-AUC.
- In cross-validation, Logistic Regression, KNN and Random Forest all got about 85% accuracy.

**Strengths:**
- The project covers all the steps, from the problem to the evaluation.
- I found the hidden missing values (the zeros) and fixed them.
- I fitted the scaler only on the training data, so the test result is fair.
- I checked the models with five metrics, not only accuracy.
- The best model is simple and its recall is high.

**Limitations:**
- The dataset is small (918 patients).
- 19% of the cholesterol values were missing and I filled all of them with the same median.
- I calculated the median before the split, which is a small data leakage.
- 79% of the patients are male.
- The data is old and from the USA and Europe, not from Bangladesh.
- The best model still missed 9 of the 102 heart disease patients.

**Future improvements:**
- Collect more data, mainly from local hospitals.
- Fill the missing cholesterol values with a better method.
- Try other models like SVM and Gradient Boosting.
- Tune the hyperparameters of all the models.
- Make a simple web app for doctors.

**Conclusion:** I used the Heart Failure Prediction dataset to predict heart disease. I cleaned the data, studied it with charts, prepared the features and trained four models. Logistic Regression gave the best result. A model like this can help a hospital to screen patients quickly, but the final decision must always be taken by a doctor.

**References:**
1. Fedesoriano. Heart Failure Prediction Dataset. Kaggle, 2021. https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
2. World Health Organization. Cardiovascular diseases (CVDs), fact sheet.
3. Pedregosa et al. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 2011.
```
