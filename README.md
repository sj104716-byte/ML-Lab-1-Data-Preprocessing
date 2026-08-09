# ML Lab-2 : Linear Regression Model for House Price Prediction

**Name:** Soumya Jaiswal &nbsp;|&nbsp; **Section:** B &nbsp;|&nbsp; **Roll No:** 10 &nbsp;|&nbsp; **Batch:** B1

Ramdeobaba University, Nagpur — Department of Artificial Intelligence and Machine Learning

---

## Aim

To develop a Linear Regression model for predicting house prices using a suitable dataset.

## Files in this repository

| File | Description |
|---|---|
| [`Linear-Regression.ipynb`](Linear-Regression.ipynb) | **Main notebook — all 29 steps with outputs and plots already executed.** Opens directly in the browser. |
| `usa_housing_regression.py` | The same pipeline as a plain Python script. |
| `USA_Housing.csv` | The dataset used (5000 rows, 7 columns). |

## Steps applied

1. Loaded the USA Housing dataset and inspected it with `head()`, `tail()`, `shape`, `info()`, `describe()`
2. Checked for missing values and dropped the text `Address` column
3. Visualised Income, Price and their correlation with box plots, histograms and a heatmap
4. Built a **Simple Linear Regression** model (`Price ~ Avg. Area Income`) — R² ≈ 0.45
5. Built a **Multiple Linear Regression** model using all 5 numeric features — R² ≈ 0.92
6. Tuned **Ridge Regression** with `GridSearchCV`
7. Tuned **Lasso Regression** with `GridSearchCV`
8. Used the trained model to predict the price of a new house

## Result

| Model | R² Score |
|---|---|
| Simple Linear Regression (Income only) | 0.447 |
| Multiple Linear Regression (all features) | 0.922 |
| Ridge Regression (tuned, alpha = 1) | 0.922 |
| Lasso Regression (tuned) | 0.922 |

## How to run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook Linear-Regression.ipynb
```
