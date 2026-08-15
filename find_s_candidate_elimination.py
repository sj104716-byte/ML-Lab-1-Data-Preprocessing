import pandas as pd
data=pd.read_csv("enjoy.csv")
data

X = data.iloc[:,:-1].values
y=data.iloc[:,-1].values

import numpy as np

def find_s(X,y):
    hypothesis=None
    print("Initial Hypothesis :",hypothesis)
    for i in range(len(X)):
        if y[i]=="Yes":
            if hypothesis is None:
                hypothesis=X[i].copy()
            else:
                for j in range(len(hypothesis)):
                    if hypothesis[j] !=X[i][j]:
                        hypothesis[j]="?"
        print(f"\nAfter Training Example {i+1}")
        print(hypothesis)
    return hypothesis

final_hypothesis=find_s(X,y)
print("\nFinal Hypothesis")
print(final_hypothesis)


def candidate_elimination(concepts,target):
    specific=concepts[0].copy()
    general=[["?" for i in range(len(specific))]]
    print("\nInitial S:",specific)
    print("Initial G:",general)
    for i,h in enumerate(concepts):
        if target[i]=="Yes":
            for x in range(len(specific)):
                if h[x]!=specific[x]:
                    specific[x]="?"

        else:
            for x in range(len(specific)):
                if h[x]!=specific[x]:
                    general.append(["?" if j!=x else specific[x] for j in range(len(specific))])

        print("\nExample",i+1)
        print("S=",specific)
        print("G=",general)

    return specific,general

S,G=candidate_elimination(X,y)

print("\nFinal Specific Boundary")
print(S)
print("\nFinal General Boundary")
for g in G:
    print(g)
