import numpy as np
import pandas as pd
from decimal import Decimal, getcontext

getcontext().prec = 60

# calculate information gain and return the best feature to split on

def main():
    filepath = './Decision_Tree/testDataA4/continue2.in'

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
        if(len(targets) == 0 or count == 0):
            product += 0
        else:
            product += (-1 * (count / len(targets)) * np.log2(count / len(targets)))
    return product

def infoAttributeD(targets, inputs, attributes, labels):
    continuous = False
    continuousAttributes = []
    continuousIndices = []
    counts = []
    countsGT = []
    countsLT = []
    partitionedCountsLT = []
    partitionedCountsGT = []
    bestSplitAttributeIndex: int = -1
    bestSplitAttribute: str = ''
    currentInfoGain = Decimal(0)
    bestInfoGain = Decimal(-1)
    targetedCounts = []
    indexer = 0
    indexer2 = 0
    indexerLT = 0
    indexerGT = 0
    indexer2LT = 0
    indexer2GT = 0
    product = Decimal(0)
    possibleSplits = []
    
    # Attribute is the line in the file, ex: ['age', '0-30', '31-60', '61-100']
    # Value is the value of the attribute, ex: '0-30', and we skip the first value in the attribute list 
    #       b/c it is the attribute name
    # Target is the target value, ex: 'yes' or 'no', and is there to compare to the inputs for the coming math
    #       essentially I need the amount of 'yes's and 'no's for each value of the attribute, and the 
    #       target loop is for that.
    for i, attribute in enumerate(attributes):
        for j, value in enumerate(attribute):
            if(j == 0):
                continue
            elif(value == 'continuous'):
                continuous = True
                continuousColumn = inputs[:, i]
                continuousColumn = np.sort(continuousColumn)
                continuousColumn = np.unique(continuousColumn)
                attribute.pop(1)
                print(attribute)

                for m in range(len(continuousColumn) - 1):
                    attribute.append((float(continuousColumn[m]) + float(continuousColumn[m+1])) / 2)
                value = attribute[j]
                print(value)
                print(attribute)
                

            for k, target in enumerate(targets):
                if continuous:
                    if value 
                if value == inputs[k, i]:
                    indexer += 1
                    if target == labels[0]:
                        indexer2 += 1

            # Store the amount of 'yes's and 'no's for each value of the attribute and reset the counters.
            targetedCounts.append(indexer2)            
            counts.append(indexer)
            indexer = 0
            indexer2 = 0

        # If the attribute is continuous, we need to calculate the information gain differently
        # if(continuous):

            # for split in possibleSplits:
            #     for n, target in enumerate(targets):
            #         if float(inputs[n, i]) < split:
            #             indexerLT += 1
            #             if target == labels[0]:
            #                 indexer2LT += 1
            #         elif float(inputs[n, i]) >= split:
            #             indexerGT += 1
            #             if target == labels[0]:
            #                 indexer2GT += 1
            #     countsLT.append(indexerLT)
            #     countsGT.append(indexerGT)
            #     partitionedCountsLT.append(indexer2LT)
            #     partitionedCountsGT.append(indexer2GT)
            #     indexerLT = 0
            #     indexerGT = 0
            #     indexer2LT = 0
            #     indexer2GT = 0

            # for l, count in enumerate(countsLT):


        # The math that actually calculates the information gain
        for l, count in enumerate(counts):
            if(len(targets) == 0):
                product += 0
            elif (count - targetedCounts[l] == 0 or targetedCounts[l] == 0):
                product += 0
            else:
                product = (product + Decimal(((count/len(targets)) * (-1 * ((targetedCounts[l] / count) * np.log2(targetedCounts[l] / count) + (((count - targetedCounts[l]) / count) * np.log2((count - targetedCounts[l]) / count)))))))
        
        infod = Decimal(infoD(targets, labels))
        currentInfoGain = infod - product
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
    splitAttributeIndex = -1000000

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
    
    rowCount = len(inputs)

    labelCount = [0] * len(labels)

    for i in range(rowCount):
        label = targets[i]
        labelIDX = labels.index(label)
        labelCount[labelIDX] += 1

    all_same = all(target == targets[0] for target in targets)
    if(all_same):
        if(len(targets) != 0):
            print("\t" * (depth) + targets[0])
        else:
            print("\t" * (depth) + "No data")
        return
    if len(attributes) == 0:
        print("\t" * (depth) + labels[labelCount.index(max(labelCount))])
        return

    bestSplit, bestSplitIndex = infoAttributeD(targets, inputs, attributes, labels)
    newInputs, newTargets, newAttributes = splitOnAttribute(bestSplit, targets, inputs, attributes)

    for i, input in enumerate(newInputs):
        print("\t" * (depth) + bestSplit + " = " + attributes[bestSplitIndex][i+1] + ":")

        generateDecisionTree(newTargets[i], input, newAttributes, labels, depth+1)



if __name__ == "__main__":
    main()