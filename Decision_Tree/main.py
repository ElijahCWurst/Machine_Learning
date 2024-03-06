import numpy as np
import pandas as pd

# calculate information gain and return the best feature to split on

def loadData(filepath: str):
    attributeCount = 0
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

    return attributes, labels, data

def infoD(targets: list[int], labels) -> float:
    # count how many times each label appears in the targets
    counts = []
    product: float = 0
    for i, label in enumerate(labels):
        i = 0
        for target in targets:
            if label == target:
                i += 1
        counts.append(i)
    print(counts[0])
    print(counts[1])
    for count in counts:
        if(len(targets) == 0):
            product += 0
        else:
            product += (-1 * (count / len(targets)) * np.log2(count / len(targets)))
    return product

def infoAttributeD(targets: list[int], inputs, attributes, labels) -> float:
    counts = []
    indexer = 0
    for i, attribute in enumerate(attributes):
        for j, value in enumerate(attribute):
            for k, target in enumerate(targets):
                if value == inputs[k, i]:
                    indexer += 1
            counts.append(indexer)
            indexer = 0
        counts.pop(0)
        print(attribute)
        print(inputs[:, i])
        print(counts)
        counts.clear()
    return -1




filepath = './Decision_Tree/testDataA4/golf.in'

attributes, labels, trainingdatastr = loadData(filepath)

trainingdata: np.ndarray[int, np.dtype[np.int_]] = np.array(trainingdatastr)

inputs = trainingdata[:, :-1]
targetsNP = trainingdata[:, -1]

targetsList = targetsNP.tolist()

# remove the first column from attributes
# attributes = [attr[1:] for attr in attributes]

# print(infoD(targetsList, labels))
infoAttributeD(targetsList, inputs, attributes, labels)

# print(pd.DataFrame(trainingdata))
# print(pd.DataFrame(inputs))
# print(pd.DataFrame(targetsNP))
# print(pd.DataFrame(attributes))
# print(pd.DataFrame(labels))

