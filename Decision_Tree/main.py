import numpy as np
import pandas as pd

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

filepath = './Decision_Tree/testDataA4/golf.in'

# print(attributeCount)
attributes: list[list[str]] = []
labels: list[str] = []
data: list[list[str]] = []


for i, line in enumerate(open(filepath)):
    if(i == 0):
        attributeCount = int(line)
    elif(i <= attributeCount):
        attributes.append(line.split())
    elif(i == attributeCount + 1):
        labels = line.split()
        labels.pop(0)
    else:
        data.append(line.split())

print(pd.DataFrame(attributes))
print(pd.DataFrame(labels))
print(pd.DataFrame(data))