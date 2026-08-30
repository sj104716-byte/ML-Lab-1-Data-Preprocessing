# ML Lab-4 : Decision Tree Classifier for Wine Quality Prediction

**Name:** Soumya Jaiswal &nbsp;|&nbsp; **Section:** B &nbsp;|&nbsp; **Roll No:** 10 &nbsp;|&nbsp; **Batch:** B1

Ramdeobaba University, Nagpur — Department of Artificial Intelligence and Machine Learning

---

## Aim

To develop a Decision Tree classifier that predicts wine quality from its physicochemical
properties, evaluate it with accuracy, precision, recall, F1-score, a confusion matrix and
ROC-AUC curves, and then improve it with GridSearchCV hyperparameter tuning validated by
cross-validation, learning curves and validation curves.

## Files in this repository

| File | Description |
|---|---|
| [`c2-b1-10-soumya-prac4-ml.ipynb`](c2-b1-10-soumya-prac4-ml.ipynb) | **Main notebook — every step with outputs and all 51 plots already executed.** Opens directly in the browser. |
| `wine_quality_decision_tree.py` | The same pipeline as a plain Python script. |
| `WineQT.csv` | The dataset used (1599 rows, 13 columns). |

## Steps applied

1. Loaded the WineQT dataset and inspected it with `head()`, `tail()`, `shape`, `info()` and `describe()`
2. Checked for missing values and duplicate rows — none found, all 1599 rows kept
3. **Univariate analysis** — histograms with KDE and box plots for every column, plus a count plot of the quality label
4. **Bivariate analysis** — box plots of all 11 physicochemical features against `quality`, and scatter plots for `alcohol` and `volatile acidity`
5. **Multivariate analysis** — correlation heat-map and a pair plot of the four strongest features
6. Binned `quality` into Low / Medium / High groups and compared alcohol content across them
7. Split the data 80 / 20 with `stratify=y` so every quality score keeps its proportion (1279 train / 320 test)
8. Built a **Decision Tree** with `criterion='entropy'` (information gain) and trained it
9. Scored it with accuracy, weighted precision / recall / F1 and a `classification_report`
10. Plotted the **confusion matrix** as a heat-map
11. Visualised the tree — the top 3 levels for readability and then the full tree
12. Plotted **one-vs-rest ROC curves** for each quality class and the overall weighted ROC-AUC
13. Tuned the tree with **GridSearchCV** over 960 combinations of `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features` and `ccp_alpha` (5-fold CV)
14. Validated the tuned tree with **10-fold cross-validation**, a **learning curve** and a **validation curve** over `max_depth`

## Dataset

| Item | Value |
|---|---|
| Rows × columns | 1599 × 13 |
| Features | 11 physicochemical + `Id` |
| Target | `quality` (scores 3 – 8) |
| Class counts | 3 → 10, 4 → 53, 5 → 681, 6 → 638, 7 → 199, 8 → 18 |

## Result

| Metric | Score |
|---|---|
| Training accuracy (unpruned tree) | 1.000 |
| Testing accuracy (unpruned tree) | 0.584 |
| Weighted precision | 0.591 |
| Weighted recall | 0.584 |
| Weighted F1-score | 0.587 |
| Overall ROC-AUC (one-vs-rest, weighted) | 0.683 |
| Best GridSearchCV cross-validation accuracy | 0.600 |
| Testing accuracy after tuning | 0.584 |
| 10-fold cross-validation average | 0.474 (std 0.042) |
| Tree depth / leaves | 19 / 290 |

**Best parameters found by GridSearchCV**

```python
{'ccp_alpha': 0, 'criterion': 'entropy', 'max_depth': 20,
 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 2}
```

**Observation** — the unpruned tree reaches 100% training accuracy but only 58.4% on the test set,
a textbook case of overfitting (depth 19, 290 leaves). The classes are heavily imbalanced: quality
5 and 6 dominate the data, so the tree predicts those two well (F1 of 0.65 and 0.59) while rare
scores such as 3 and 8 are almost never predicted correctly. The validation curve shows accuracy
peaking at a shallow `max_depth` and then flattening, confirming that extra depth only memorises
the training data.

## How to run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook c2-b1-10-soumya-prac4-ml.ipynb
```

Or run the script version directly:

```bash
python wine_quality_decision_tree.py
```
