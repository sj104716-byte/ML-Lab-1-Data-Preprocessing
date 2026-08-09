"""
Lab-1 : Applying Data Preprocessing Techniques to the Titanic Dataset
Student : Soumya   |   Section B  |  Roll No. 10  |  Batch B1
Goal    : Clean, transform and scale the Titanic dataset so that it becomes
          suitable for training a machine learning classifier.
"""

import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

matplotlib.use("Agg")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 25)


def banner(step, title):
    print(f"STEP {step} : {title}")


# ======================================================================
# STEP 1 : READ THE DATA
# ======================================================================
banner(1, "READING THE TITANIC DATA FILE")
data = pd.read_csv("titanic.csv")
print("Rows and columns present in the file :", data.shape)
print("Column names :", ", ".join(data.columns))
print("\nSample records from the top of the file:\n")
print(data.head())

# ======================================================================
# STEP 2 : EXAMINE DATA TYPES AND SPREAD
# ======================================================================
banner(2, "EXAMINING DATA TYPES AND SPREAD")
data.info()
print("\nSpread of the numeric columns:\n")
print(data.describe().T.round(2))

# ======================================================================
# STEP 3 : LOCATE THE GAPS IN THE DATA
# ======================================================================
banner(3, "LOCATING THE GAPS IN THE DATA")
gaps = data.isnull().sum()
gaps = gaps[gaps > 0].sort_values(ascending=False)
report = pd.DataFrame({"Blanks": gaps, "Percent": (gaps / len(data) * 100).round(2)})
print(report)
print("\nRepeated rows found :", data.duplicated().sum())

plt.figure(figsize=(6.5, 3.6), facecolor="white")
bars = plt.bar(report.index, report.Percent, color="#5b8ff9", width=0.5)
for b, v in zip(bars, report.Percent):
    plt.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v}%", ha="center", fontsize=9)
plt.ylabel("Percentage of blank values")
plt.ylim(0, 90)
plt.title("Percentage of Blank Values per Column")
plt.tight_layout()
plt.savefig("out_s/fig_missing.png", dpi=130, facecolor="white")
plt.close()

# ======================================================================
# STEP 4 : FILL IN THE GAPS
# ======================================================================
banner(4, "FILLING IN THE GAPS")
age_lookup = data.groupby(["Pclass", "Sex"])["Age"].median()
print("Median age reference table used for filling:\n")
print(age_lookup.round(1))
data["Age"] = data.groupby(["Pclass", "Sex"])["Age"].transform(lambda s: s.fillna(s.median()))

top_port = data["Embarked"].value_counts().idxmax()
data["Embarked"] = data["Embarked"].fillna(top_port)
data["Has_Cabin"] = np.where(data["Cabin"].isnull(), 0, 1)

print(f"\nEmbarked blanks replaced with the most frequent port : {top_port}")
print("Cabin turned into the indicator column 'Has_Cabin' instead of being filled.")
print("Blanks remaining in Age / Embarked :",
      int(data["Age"].isnull().sum()), "/", int(data["Embarked"].isnull().sum()))

# ======================================================================
# STEP 5 : REMOVE COLUMNS THAT CANNOT HELP THE MODEL
# ======================================================================
banner(5, "REMOVING COLUMNS THAT CANNOT HELP THE MODEL")
junk = ["PassengerId", "Name", "Ticket", "Cabin"]
data.drop(columns=junk, inplace=True)
print("Columns removed :", junk)
print("Size of the table now :", data.shape)
print("Columns retained :", list(data.columns))

# ======================================================================
# STEP 6 : LIMIT THE EXTREME VALUES
# ======================================================================
banner(6, "LIMITING THE EXTREME VALUES WITH THE IQR RULE")


def iqr_limits(series):
    q1, q3 = series.quantile([0.25, 0.75])
    spread = q3 - q1
    return q1 - 1.5 * spread, q3 + 1.5 * spread, q1, q3, spread


plt.figure(figsize=(7.5, 3.4), facecolor="white")
plt.subplot(1, 2, 1)
sns.boxplot(y=data["Fare"], color="#f6c667", width=0.35)
plt.title("Fare : original values")

for col in ["Age", "Fare"]:
    lo, hi, q1, q3, spread = iqr_limits(data[col])
    flagged = int(((data[col] < lo) | (data[col] > hi)).sum())
    data[col] = data[col].clip(lo, hi)
    print(f"{col:<5} Q1={q1:7.2f} Q3={q3:7.2f} IQR={spread:7.2f} "
          f"allowed=[{lo:7.2f} , {hi:7.2f}] extreme values pulled in = {flagged}")

plt.subplot(1, 2, 2)
sns.boxplot(y=data["Fare"], color="#7ecba1", width=0.35)
plt.title("Fare : after limiting")
plt.tight_layout()
plt.savefig("out_s/fig_outlier.png", dpi=130, facecolor="white")
plt.close()

print("\nRange of Age and Fare once the limits are applied:\n")
print(data[["Age", "Fare"]].agg(["min", "max", "mean", "std"]).round(2))

# ======================================================================
# STEP 7 : BUILD NEW COLUMNS FROM THE OLD ONES
# ======================================================================
banner(7, "BUILDING NEW COLUMNS FROM THE OLD ONES")
data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
data["IsAlone"] = np.where(data["FamilySize"] == 1, 1, 0)
print("FamilySize = SibSp + Parch + 1  (the passenger is counted as well)")
print("IsAlone    = 1 when FamilySize equals 1, otherwise 0\n")
print("Distribution of the new IsAlone column:")
print(data["IsAlone"].value_counts().rename({0: "With family", 1: "Travelling alone"}))
print("\nSample of the derived columns:\n")
print(data[["SibSp", "Parch", "FamilySize", "IsAlone"]].head(6))

# ======================================================================
# STEP 8 : TURN TEXT COLUMNS INTO NUMBERS
# ======================================================================
banner(8, "TURNING TEXT COLUMNS INTO NUMBERS")
encoder = LabelEncoder()
data["Sex"] = encoder.fit_transform(data["Sex"])
print("LabelEncoder mapping for Sex :", dict(zip(encoder.classes_, range(len(encoder.classes_)))))

data = pd.get_dummies(data, columns=["Embarked"], prefix="Port", drop_first=True, dtype=int)
print("Embarked expanded into :", [c for c in data.columns if c.startswith("Port")])
print("The first category was dropped so that the dummy columns stay independent.\n")
print(data.head())

# ======================================================================
# STEP 9 : BRING THE NUMBERS TO A COMMON SCALE
# ======================================================================
banner(9, "BRINGING THE NUMBERS TO A COMMON SCALE")
to_scale = ["Age", "Fare", "FamilySize"]
print("Values before scaling:\n")
print(data[to_scale].head(6).round(2))

data[to_scale] = StandardScaler().fit_transform(data[to_scale])

print("\nValues after scaling:\n")
print(data[to_scale].head(6).round(2))
print("\nMean and standard deviation of the scaled columns:")
print(data[to_scale].agg(["mean", "std"]).round(2))

# ======================================================================
# STEP 10 : SEPARATE THE DATA AND CONFIRM IT IS READY
# ======================================================================
banner(10, "SEPARATING THE DATA AND CONFIRMING IT IS READY")
X = data.drop(columns="Survived")
y = data["Survived"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=7, stratify=y)

print("Inputs :", X.shape, "  Target :", y.shape)
print("Used for training :", X_tr.shape[0], "rows    Kept for testing :", X_te.shape[0], "rows")
print("Blanks anywhere in the final table :", int(data.isnull().sum().sum()))
print("Column types present :", sorted(set(map(str, data.dtypes))))

clf = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
score = accuracy_score(y_te, clf.predict(X_te))
print(f"\nTrial run - Logistic Regression scored {score*100:.2f}% on the unseen rows.")

data.to_csv("out_s/titanic_ready.csv", index=False)
print("Processed table written to 'titanic_ready.csv' with size", data.shape)
print("\nFirst rows of the processed table:\n")
print(data.head())

plt.figure(figsize=(7.2, 5.0), facecolor="white")
sns.heatmap(data.corr().round(2), annot=True, fmt=".2f", cmap="YlGnBu",
            annot_kws={"size": 7}, linewidths=0.4, cbar_kws={"shrink": 0.8})
plt.title("How the Processed Columns Relate to One Another")
plt.tight_layout()
plt.savefig("out_s/fig_corr.png", dpi=130, facecolor="white")
plt.close()
