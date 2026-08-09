# ML Lab-1 : Data Preprocessing on the Titanic Dataset

**Name:** Soumya Jaiswal &nbsp;|&nbsp; **Section:** B &nbsp;|&nbsp; **Roll No:** 10 &nbsp;|&nbsp; **Batch:** B1

Ramdeobaba University, Nagpur — Department of Artificial Intelligence and Machine Learning

---

## Aim

To study and apply Data Preprocessing techniques on the given dataset. Prepare a Titanic dataset for
training with a machine learning algorithm by applying suitable data preprocessing techniques.

## Files in this repository

| File | Description |
|---|---|
| [`Data-Preprocessing.ipynb`](Data-Preprocessing.ipynb) | **Main notebook — all 29 steps with their outputs and plots already executed.** Opens directly in the browser. |
| `titanic_preprocessing_soumya.py` | The same code as a plain Python script. |
| `titanic.csv` | The dataset used (891 rows, 12 columns). |

## Preprocessing steps applied

1. Loaded the dataset and inspected it with `head()`, `tail()`, `shape`, `info()` and `describe()`
2. Counted the missing values — Age 177, Cabin 687, Embarked 2
3. Label encoded `Embarked` (S→0, C→1, Q→2) and `Sex` (male→0, female→1)
4. Filled `Age` with its median and `Embarked` with its mode
5. Dropped `PassengerId`, `Name`, `Ticket` and `Cabin`
6. Detected and removed 116 duplicate rows
7. Visualised the data — box plots, histograms, count plots, scatter plot and a correlation heatmap
8. Separated the features `X` from the target `y`
9. Standardized `Age` and `Fare` using `StandardScaler`
10. Split the data 80:20 into training and testing sets

## Result

| Item | Value |
|---|---|
| Original dataset | 891 rows × 12 columns |
| Duplicates removed | 116 |
| Final dataset | 775 rows × 8 columns |
| Training set | 620 rows × 7 features |
| Testing set | 155 rows × 7 features |

## How to run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook Data-Preprocessing.ipynb
```
