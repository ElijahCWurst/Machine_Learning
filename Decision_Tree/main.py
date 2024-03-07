import numpy as np
import pandas as pd
import statistics
# calculate information gain and return the best feature to split on

def main():
    filepath = './Decision_Tree/testDataA4/restaurantDecisionTree.in'

    attributes, labels, trainingdatastr = loadData(filepath)

    trainingdata = np.array(trainingdatastr)

    inputs = trainingdata[:, :-1]
    targetsNP = trainingdata[:, -1]

    targetsList = targetsNP.tolist()

    # bestSplitAttribute = infoAttributeD(targetsList, inputs, attributes, labels)
    # print(bestSplitAttribute)


    generateDecisionTree(targetsList, inputs, attributes, labels)


    # print(pd.DataFrame(trainingdata))
    # print(pd.DataFrame(inputs))
    # print(pd.DataFrame(targetsNP))
    # print(pd.DataFrame(attributes))
    # print(pd.DataFrame(labels))





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

def infoAttributeD(targets, inputs, attributes, labels):

    counts = []
    bestSplitAttributeIndex: int = -1
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
            bestSplitAttributeIndex = i

        product = 0
        targetedCounts.clear()
        counts.clear()
    return bestSplitAttribute, bestSplitAttributeIndex

def splitOnAttribute(splitAttribute, targets, inputs, attributes):
    splitInput = []
    splitTarget = []
    splitInputList = []
    splitTargetList = []

    for attribute in attributes:
        if attribute[0] == splitAttribute:
            splitAttributeIndex = attributes.index(attribute)
            break
    for j, value in enumerate(attributes[splitAttributeIndex]):
        if j == 0:
            continue
        for i, input in enumerate(inputs):
            if input[splitAttributeIndex] == attributes[splitAttributeIndex][j]:
                splitInput.append(input)
                splitTarget.append(targets[i])
        splitInputList.append(np.array(splitInput))
        splitTargetList.append(np.array(splitTarget))
        splitInput.clear()
        splitTarget.clear()
    
    newAttributes = attributes.copy()
    newAttributes.pop(splitAttributeIndex)
    
    # remove the split attribute from the splitInputList
    for i, input in enumerate(splitInputList):
        if(input.size == 0):
            continue
        splitInputList[i] = np.delete(input, splitAttributeIndex, 1)
        
    return splitInputList, splitTargetList, newAttributes

def generateDecisionTree(targets, inputs, attributes, labels, depth=0):
    
    all_same = all(target == targets[0] for target in targets)
    if(all_same):
        print("    " * (depth) + targets[0])
        return
    if len(attributes) == 0:
        print("No attributes")
        return

    bestSplit, bestSplitIndex = infoAttributeD(targets, inputs, attributes, labels)
    newInputs, newTargets, newAttributes = splitOnAttribute(bestSplit, targets, inputs, attributes)

    for i, input in enumerate(newInputs):
        print("    " * (depth) + bestSplit + " = " + attributes[bestSplitIndex][i+1] + ":")

        generateDecisionTree(newTargets[i], input, newAttributes, labels, depth+1)



if __name__ == "__main__":
    main()