"""
Lab-1 : Data Preprocessing on the Titanic Dataset
Name    : Soumya
Section : B          Roll No : 10          Batch : B1
Aim     : To study and apply Data Preprocessing techniques on the given dataset
          and prepare the Titanic dataset for training with a machine learning
          algorithm by applying suitable data preprocessing techniques.
"""

# Output 1 : Importing the required libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

# Output 2 : Loading the dataset
df = pd.read_csv("titanic.csv")

# Output 3 : First five records
print(df.head())

# Output 4 : Last five records
print(df.tail())

# Output 5 : Shape of the dataset
print(df.shape)

# Output 6 : Structure of the dataset
df.info()

# Output 7 : Statistical summary
print(df.describe())

# Output 8 : Count of missing values
print(df.isnull().sum())

# Output 9 : Unique values in the Embarked column
print(df["Embarked"].unique())

# Output 10 : Label encoding the Embarked column
df["Embarked"] = df["Embarked"].replace("S", 0)
df["Embarked"] = df["Embarked"].replace("C", 1)
df["Embarked"] = df["Embarked"].replace("Q", 2)
print(df.head())

# Output 11 : Unique values in the Sex column
print(df["Sex"].unique())

# Output 12 : Label encoding the Sex column
df["Sex"] = df["Sex"].replace("male", 0)
df["Sex"] = df["Sex"].replace("female", 1)
print(df.head())

# Output 13 : Handling missing values and dropping irrelevant columns
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df.drop("PassengerId", axis=1, inplace=True)
df.drop("Name", axis=1, inplace=True)
df.drop("Ticket", axis=1, inplace=True)
df.drop("Cabin", axis=1, inplace=True)
print(df.head())

# Output 14 : Counting duplicate rows
print(df.duplicated().sum())

# Output 15 : Removing duplicate rows
df.drop_duplicates(inplace=True)

# Output 16 : Boxplot of Age
sns.boxplot(x=df["Age"])
plt.show()

# Output 17 : Boxplot of Fare
sns.boxplot(x=df["Fare"])
plt.show()

# Output 18 : Boxplot of Parch
sns.boxplot(x=df["Parch"])
plt.show()

# Output 19 : Distribution of Age
sns.histplot(df["Age"], kde=True)
plt.show()

# Output 20 : Count of survivors
sns.countplot(x="Survived", data=df)
plt.show()

# Output 21 : Gender against survival
sns.countplot(x="Sex", hue="Survived", data=df)
plt.show()

# Output 22 : Age against Fare
sns.scatterplot(x="Age", y="Fare", data=df)
plt.show()

# Output 23 : Correlation heatmap
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=['number'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='inferno')
plt.show()

# Output 24 : Passenger class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x="Pclass", data=df)
plt.title("passenger class distribution")
plt.xlabel("passenger class")
plt.ylabel("count")
plt.show()

# Output 25 : Passengers by embarked port
plt.figure(figsize=(6, 4))
sns.countplot(x="Embarked", data=df)
plt.title("passenger by embarked port")
plt.show()

# Output 26 : Fare distribution
plt.figure(figsize=(6, 4))
plt.hist(df["Fare"], bins=20)
plt.title("fare distribution")
plt.xlabel("fare")
plt.ylabel("frequemcy")
plt.show()

# Output 27 : Separating features and target
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Output 28 : Feature scaling
scaler = StandardScaler()
X[["Age", "Fare"]] = scaler.fit_transform(X[["Age", "Fare"]])
print(X.head(15))

# Output 29 : Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=4)
print("training data:", X_train.shape)
print("testing data:", X_test.shape)
print(X_train.head())
