# ML Lab 3: Find-S and Candidate Elimination

Implements and compares two concept-learning algorithms on the EnjoySport dataset:

- **Find-S** — finds the maximally specific hypothesis consistent with the positive training examples.
- **Candidate Elimination** — maintains the specific boundary (S) and general boundary (G) of the version space using both positive and negative examples.

## Files

- `find_s_candidate_elimination.py` — Python implementation of both algorithms
- `enjoy.csv` — EnjoySport training dataset (10 examples, 6 attributes + target label)

## Dataset

Standard EnjoySport concept-learning dataset (Tom Mitchell, *Machine Learning*), with attributes Sky, AirTemp, Humidity, Wind, Water, Forecast, and target EnjoySport (Yes/No).

## Result

Both algorithms converge to the same specific hypothesis:

```
['Sunny', 'Warm', '?', '?', '?', '?']
```

meaning only `Sky = Sunny` and `AirTemp = Warm` are required for `EnjoySport = Yes`.
