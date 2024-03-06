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
    for count in counts:
        if(len(targets) == 0):
            product += 0
        else:
            product += (-1 * (count / len(targets)) * np.log2(count / len(targets)))
    return product

def infoAttributeD(targets: list[int], inputs, attributes, labels) -> str:

    counts = []
    bestSplitAttribute: str = ''
    currentInfoGain: float = -1
    bestInfoGain: float = -1
    targetedCounts = []
    indexer = 0
    indexer2 = 0
    product: float = 0

    for i, attribute in enumerate(attributes):
        for j, value in enumerate(attribute):
            if(j == 0):
                continue
            for k, target in enumerate(targets):
                if value == inputs[k, i]:
                    indexer += 1
                    if target == labels[0]:
                        indexer2 += 1


            targetedCounts.append(indexer2)            
            counts.append(indexer)
            indexer = 0
            indexer2 = 0

        
        for l, count in enumerate(counts):
            if(len(targets) == 0):
                product += 0
            elif (count - targetedCounts[l] == 0 or targetedCounts[l] == 0):
                product += 0
            else:
                product = (product + ((count/len(targets)) * (-1 * ((targetedCounts[l] / count) * np.log2(targetedCounts[l] / count) + (((count - targetedCounts[l]) / count) * np.log2((count - targetedCounts[l]) / count))))))
        
        currentInfoGain = infoD(targets, labels) - product
        if(currentInfoGain > bestInfoGain):
            bestInfoGain = currentInfoGain
            bestSplitAttribute = attribute[0]

        product = 0
        targetedCounts.clear()
        counts.clear()
    return bestSplitAttribute

def split_data(X: np.ndarray[int, np.dtype[np.int_]], y: list[int], feature: int, value: int) -> tuple[np.ndarray[int, np.dtype[np.int_]], list, np.ndarray[int, np.dtype[np.int_]], list]:
    true_indices = np.where(X[:, feature] <= value)[0]
    false_indices = np.where(X[:, feature] > value)[0]
    true_X, true_y = X[true_indices], y[true_indices]
    false_X, false_y = X[false_indices], y[false_indices]
    return true_X, true_y, false_X, false_y # type: ignore

def generateDecisionTree(X: np.ndarray[int, np.dtype[np.int_]], y: list[int], attributes: list[list[str]], labels: list[str]) -> dict:
    # if all labels are the same, return a leaf node with that label
    if len(set(y)) == 1:
        return {'label': y[0]}
    # if no features are left, return a leaf node with the most common label
    if len(attributes) == 0:
        most_common_label = max(set(y), key=y.count)
        return {'label': most_common_label}
    # find the best feature to split on
    best_feature = infoAttributeD(y, X, attributes, labels)
    # create a new decision tree node with the best feature
    tree = {'feature': best_feature, 'branches': {}}
    # remove the best feature from the list of attributes
    best_feature_index = [i for i, attribute in enumerate(attributes) if attribute[0] == best_feature][0]
    attributes = attributes[:best_feature_index] + attributes[best_feature_index + 1:]
    # for each possible value of the best feature, create a new branch
    for value in set(X[:, best_feature_index]):
        true_X, true_y, false_X, false_y = split_data(X, y, best_feature_index, value)
        tree['branches'][value] = generateDecisionTree(true_X, true_y, attributes, labels)
        tree['branches'][value] = generateDecisionTree(false_X, false_y, attributes, labels)
    return tree

filepath = './Decision_Tree/testDataA4/restaurantDecisionTree.in'

attributes, labels, trainingdatastr = loadData(filepath)

trainingdata: np.ndarray[int, np.dtype[np.int_]] = np.array(trainingdatastr)

inputs = trainingdata[:, :-1]
targetsNP = trainingdata[:, -1]

targetsList = targetsNP.tolist()

bestSplitAttribute = infoAttributeD(targetsList, inputs, attributes, labels)
print(bestSplitAttribute)

# print(pd.DataFrame(trainingdata))
# print(pd.DataFrame(inputs))
# print(pd.DataFrame(targetsNP))
# print(pd.DataFrame(attributes))
# print(pd.DataFrame(labels))

