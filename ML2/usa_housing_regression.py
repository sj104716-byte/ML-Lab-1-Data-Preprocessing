import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df = pd.read_csv("USA_Housing.csv")

print(df.head(10))
print(df.tail())
print(df.shape)
df.info()
print(df.describe())
print(df.isnull().sum())

df.drop("Address", axis=1, inplace=True)
print(df.head())

sns.boxplot(x=df["Avg. Area Income"])
plt.show()

sns.boxplot(x=df["Price"])
plt.show()

sns.histplot(df["Price"], kde=True)
plt.show()

plt.figure(figsize=(6, 4))
plt.hist(df["Avg. Area Income"], bins=20)
plt.title("Average Area Income Distribution")
plt.xlabel("Average Area Income")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=["number"])
sns.heatmap(numeric_df.corr(), annot=True, cmap="inferno")
plt.show()

# Simple Linear Regression
X = df[["Avg. Area Income"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

slr_model = LinearRegression()
slr_model.fit(X_train, y_train)

y_pred = slr_model.predict(X_test)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print("Simple Linear Regression Result")
print("-------------------------------")
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("RMSE :", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R^2 Score :", r2_score(y_test, y_pred))

print("Intercept:", slr_model.intercept_)
print("Slope:", slr_model.coef_[0])

plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color="green", label="Actual Data")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Regression Line")
plt.xlabel("Average Area Income")
plt.ylabel("House Price")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()

income = 68000.0
new_data = np.array([[income]])
prediction = slr_model.predict(new_data)
print("Predicted House Price = ${:,.2f}".format(prediction[0]))

# Multiple Linear Regression
X = df.drop(["Price"], axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

mlr = LinearRegression()
mlr.fit(X_train, y_train)

coef = pd.DataFrame(mlr.coef_, X.columns, columns=["Coefficient"])
print(coef)

y_pred = mlr.predict(X_test)

print("Multiple Linear Regression Result")
print("-------------------------------")
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("RMSE :", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R^2 Score :", r2_score(y_test, y_pred))

predictions = mlr.predict(X_test)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=predictions, color="orange", label="Prediction")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red",
         linewidth=2, label="Perfect Fit")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.show()

# Hyperparameter Tuning - Ridge Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

ridge = Ridge()
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
print("Default Ridge R^2 :", r2_score(y_test, ridge_pred))

param_grid = {"alpha": [0.001, 0.01, 0.1, 1, 10, 100]}
grid_ridge = GridSearchCV(estimator=Ridge(), param_grid=param_grid, scoring="r2", cv=5)
grid_ridge.fit(X_train, y_train)

print("Best Alpha :", grid_ridge.best_params_)
print(grid_ridge.best_score_)

best_ridge = grid_ridge.best_estimator_
ridge_pred = best_ridge.predict(X_test)

print("Ridge Regression Result")
print("-------------------------------")
print("MAE :", mean_absolute_error(y_test, ridge_pred))
print("MSE :", mean_squared_error(y_test, ridge_pred))
print("RMSE :", np.sqrt(mean_squared_error(y_test, ridge_pred)))
print("R^2 Score :", r2_score(y_test, ridge_pred))

# Hyperparameter Tuning - Lasso Regression
from sklearn.linear_model import Lasso

param_grid = {"alpha": [0.001, 0.01, 0.1, 1, 10]}
grid_lasso = GridSearchCV(Lasso(max_iter=5000), param_grid, cv=5, scoring="r2")
grid_lasso.fit(X_train, y_train)

print(grid_lasso.best_params_)

lasso_pred = grid_lasso.predict(X_test)
print(r2_score(y_test, lasso_pred))

print("Lasso Regression Result")
print("-------------------------------")
print("MAE :", mean_absolute_error(y_test, lasso_pred))
print("MSE :", mean_squared_error(y_test, lasso_pred))
print("RMSE :", np.sqrt(mean_squared_error(y_test, lasso_pred)))
print("R^2 Score :", r2_score(y_test, lasso_pred))

# New Data
income = 65000.0
house_age = 5.8
rooms = 6.9
bedrooms = 3.5
population = 32000.0
test_input = np.array([[income, house_age, rooms, bedrooms, population]])
predicted_price = mlr.predict(test_input)
print("Predicted House Price: ${:,.2f}".format(predicted_price[0]))
