"""
Lab-4 : Decision Tree Classifier for Wine Quality Prediction
Author  : Soumya Jaiswal  (Section B, Roll No. 10, Batch B1)
Purpose : Build a Decision Tree classifier that predicts wine quality from the
          WineQT dataset, evaluate it with accuracy / precision / recall / F1,
          ROC-AUC and a confusion matrix, then tune it with GridSearchCV and
          validate it with cross-validation, learning and validation curves.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_score,
    learning_curve,
    train_test_split,
    validation_curve,
)
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings("ignore")
pd.set_option("display.width", 120)

RANDOM_STATE = 42

# ----------------------------------------------------------------------
# Loading and inspecting the dataset
# ----------------------------------------------------------------------
df = pd.read_csv("WineQT.csv")
print("Shape of raw dataset (rows, columns):", df.shape)
print(df.head())
print(df.tail())
df.info()
print(df.describe().round(3))
print("Missing values per column:\n", df.isnull().sum())

# ----------------------------------------------------------------------
# Removing duplicate rows
# ----------------------------------------------------------------------
print("Duplicate rows before cleaning:", df.duplicated().sum())
df = df.drop_duplicates()
print("Duplicate rows after cleaning :", df.duplicated().sum())
print("Shape after dropping duplicates:", df.shape)
print("Columns:", list(df.columns))

# ----------------------------------------------------------------------
# 1. Univariate analysis
# ----------------------------------------------------------------------
numeric_cols = df.columns

for col in numeric_cols:
    plt.figure(figsize=(7, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

for col in numeric_cols:
    plt.figure(figsize=(7, 3))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

plt.figure(figsize=(7, 4))
sns.countplot(x="quality", data=df)
plt.title("Distribution of Wine Quality")
plt.show()

print("Wine count per quality score:\n", df["quality"].value_counts().sort_index())

# ----------------------------------------------------------------------
# 2. Bivariate analysis -- every feature against the quality label
# ----------------------------------------------------------------------
features = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

for col in features:
    plt.figure(figsize=(7, 4))
    sns.boxplot(x="quality", y=col, data=df)
    plt.title(f"{col} vs Wine Quality")
    plt.show()

for col in ["alcohol", "volatile acidity"]:
    sns.scatterplot(x=col, y="quality", data=df)
    plt.title(f"{col} vs quality")
    plt.show()

# ----------------------------------------------------------------------
# 3. Multivariate analysis -- correlation heat-map and pair plot
# ----------------------------------------------------------------------
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation heatmap")
plt.show()

selected = ["alcohol", "volatile acidity", "sulphates", "citric acid", "quality"]
sns.pairplot(df[selected], hue="quality")
plt.show()

# Binning the quality score into three readable groups, for plotting only
df["quality_group"] = pd.cut(
    df["quality"], bins=[0, 5, 6, 10], labels=["Low", "Medium", "High"]
)

sns.boxplot(x="quality_group", y="alcohol", data=df)
plt.title("Alcohol Distribution Across Quality Groups")
plt.show()

# ----------------------------------------------------------------------
# Preparing the features and the target
# ----------------------------------------------------------------------
y = df["quality"]
X = df.drop(columns=["quality", "quality_group"], errors="ignore")

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Feature columns:", list(X.columns))
print("Non-numeric columns left in X:", list(X.select_dtypes(exclude="number").columns))
print(y.value_counts().sort_index())

plt.figure(figsize=(7, 4))
sns.countplot(x=y)
plt.title("Wine quality distribution")
plt.xlabel("quality")
plt.ylabel("number of wines")
plt.show()

# ----------------------------------------------------------------------
# 80 / 20 stratified train-test split
# ----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

# ----------------------------------------------------------------------
# Baseline Decision Tree -- entropy / information gain, no pruning
# ----------------------------------------------------------------------
dt_model = DecisionTreeClassifier(criterion="entropy", random_state=RANDOM_STATE)
dt_model.fit(X_train, y_train)

y_train_pred = dt_model.predict(X_train)
y_test_pred = dt_model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy :", test_accuracy)

print("Precision:", precision_score(y_test, y_test_pred, average="weighted"))
print("Recall   :", recall_score(y_test, y_test_pred, average="weighted"))
print("F1 Score :", f1_score(y_test, y_test_pred, average="weighted"))
print(classification_report(y_test, y_test_pred))

# ----------------------------------------------------------------------
# Confusion matrix
# ----------------------------------------------------------------------
cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="pink")
plt.xlabel("PREDICTED QUALITY")
plt.ylabel("ACTUAL QUALITY")
plt.title("Confusion matrix - Decision Tree")
plt.show()

# ----------------------------------------------------------------------
# Visualising the tree
# ----------------------------------------------------------------------
class_names = [str(c) for c in sorted(y.unique())]

plt.figure(figsize=(24, 12))
plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=class_names,
    filled=True,
    rounded=True,
    impurity=True,
    precision=2,
    fontsize=10,
    max_depth=3,
)
plt.title("Decision tree - top 3 levels", fontsize=18)
plt.tight_layout()
plt.show()

print("Tree Depth:", dt_model.get_depth())
print("Number of Leaves:", dt_model.get_n_leaves())

plt.figure(figsize=(25, 25))
plot_tree(
    dt_model,
    feature_names=X.columns.tolist(),
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=8,
)
plt.title("Decision Tree - Entropy")
plt.show()

# ----------------------------------------------------------------------
# ROC-AUC curve -- one-vs-rest, before tuning
# ----------------------------------------------------------------------
classes = sorted(y_test.unique())
y_test_bin = label_binarize(y_test, classes=classes)
y_score = dt_model.predict_proba(X_test)

plt.figure(figsize=(10, 7))

for i, class_label in enumerate(classes):
    fpr, tpr, thresholds = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, linewidth=2, label=f"Quality {class_label} (AUC={roc_auc:.2f})")

plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
plt.xlabel("FALSE POSITIVE RATE")
plt.ylabel("TRUE POSITIVE RATE")
plt.title("ROC CURVE - BEFORE HYPERPARAMETER TUNING")
plt.legend()
plt.grid()
plt.show()

overall_auc_before = roc_auc_score(
    y_test_bin, y_score, multi_class="ovr", average="weighted"
)
print("Overall ROC-AUC before tuning:", round(overall_auc_before, 4))

# ----------------------------------------------------------------------
# Hyperparameter tuning with GridSearchCV
# ----------------------------------------------------------------------
param_grid = {
    "criterion": ["entropy"],
    "max_depth": [3, 5, 7, 10, 20],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": [None, "sqrt", "log2"],
    "ccp_alpha": [0, 0.001, 0.005, 0.01],
}

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=RANDOM_STATE),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best Cross Validation Accuracy:", grid.best_score_)

best_tree = grid.best_estimator_

y_pred = best_tree.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy after tuning =", accuracy_score(y_test, y_pred))

# ----------------------------------------------------------------------
# Cross validation of the tuned tree
# ----------------------------------------------------------------------
scores = cross_val_score(best_tree, X, y, cv=10, scoring="accuracy")
print("10-fold scores:", scores)
print("AVERAGE ACCURACY =", scores.mean())
print("Std =", scores.std())

# ----------------------------------------------------------------------
# Learning curve -- does more training data help?
# ----------------------------------------------------------------------
train_sizes, train_scores, test_scores = learning_curve(
    best_tree, X, y, cv=5, scoring="accuracy"
)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_scores.mean(axis=1), marker="o", label="training accuracy")
plt.plot(train_sizes, test_scores.mean(axis=1), marker="o", label="validation accuracy")
plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.title("Learning curve")
plt.legend()
plt.show()

# ----------------------------------------------------------------------
# Validation curve -- accuracy against max_depth
# ----------------------------------------------------------------------
param_range = [2, 3, 4, 5, 6, 7, 8, 9, 10]

train_score, test_score = validation_curve(
    DecisionTreeClassifier(criterion="entropy", random_state=RANDOM_STATE),
    X,
    y,
    param_name="max_depth",
    param_range=param_range,
    cv=5,
)

plt.plot(param_range, test_score.mean(axis=1), label="validation")
plt.plot(param_range, train_score.mean(axis=1), label="training")
plt.xlabel("Max depth")
plt.ylabel("Accuracy")
plt.title("Validation curve")
plt.legend()
plt.show()
