import numpy as np
import pandas as pd
from decimal import Decimal, getcontext

getcontext().prec = 60

# calculate information gain and return the best feature to split on

def main():
    filepath = './Decision_Tree/testDataA4/golfc.in'

    attributes, labels, trainingdatastr = loadData(filepath)

    trainingdata = np.array(trainingdatastr)

    inputs = trainingdata[:, :-1]
    targetsNP = trainingdata[:, -1]

    targetsList = targetsNP.tolist()

    usedSplits = []

    # bestSplitAttribute = infoAttributeD(targetsList, inputs, attributes, labels)
    # print(bestSplitAttribute)


    generateDecisionTree(targetsList, inputs, attributes, labels, usedSplits)


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

def infoAttributeD(targets, inputs, attributes, labels, usedSplits):
    continuous = False
    continuousAttributes = []
    continuousIndices = []
    counts = []
    countsGT = []
    countsLT = []
    partitionedCountsLT = []
    partitionedCountsGT = []
    bestSplitAttributeIndex: int = -1
    bestContSplitAttributeIndex: int = -1
    bestSplitAttribute: str = ''
    bestContSplitAttribute: str = ''
    bestContSplit: float = -1
    currentInfoGain = Decimal(0)
    currentContInfoGain = Decimal(0)
    bestContInfoGain = Decimal(-1)
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
    alreadyCheckedLT = False
    infod = Decimal(infoD(targets, labels))
    usedSplit = ''

    
    # Attribute is the line in the file, ex: ['age', '0-30', '31-60', '61-100']
    # Value is the value of the attribute, ex: '0-30', and we skip the first value in the attribute list 
    #       b/c it is the attribute name
    # Target is the target value, ex: 'yes' or 'no', and is there to compare to the inputs for the coming math
    #       essentially I need the amount of 'yes's and 'no's for each value of the attribute, and the 
    #       target loop is for that.
    for i, attribute in enumerate(attributes):
        alreadyCheckedLT = False
        continuous = False
        for j, value in enumerate(attribute):
            if(j == 0):
                continue
            elif(value == 'continuous'):
                continuous = True

                # Get the unique values of the continuous attribute
                continuousColumn = inputs[:, i]
                continuousColumn = np.sort(continuousColumn)
                continuousColumn = np.unique(continuousColumn)
                # calculate possible splits
                for k in range(len(continuousColumn) - 1):
                    possibleSplits.append((float(continuousColumn[k]) + float(continuousColumn[k+1])) / 2)


            for k, target in enumerate(targets):
                # If the attribute is continuous, we need to check <= and > for the value of the attribute in seperate iterations
                # if continuous:
                #     if alreadyCheckedLT:
                #         if float(inputs[k, i]) > float(value):
                #             indexer += 1
                #             if target == labels[0]:
                #                 indexer2 += 1
                #     elif float(inputs[k, i]) <= float(value):

                #         indexer += 1
                #         if target == labels[0]:
                #             indexer2 += 1
                if value == inputs[k, i]:
                    indexer += 1
                    if target == labels[0]:
                        indexer2 += 1
            # if continuous:
            #     alreadyCheckedLT = True

            # Store the amount of 'yes's and 'no's for each value of the attribute and reset the counters.
            targetedCounts.append(indexer2)            
            counts.append(indexer)
            indexer = 0
            indexer2 = 0

        # If the attribute is continuous, we need to calculate the information gain differently
        if(continuous):
            for split in possibleSplits:
                if usedSplits != None:
                    if ((attribute[0] + str(split)) in usedSplits):
                        continue
                for n, target in enumerate(targets):
                    if float(inputs[n, i]) < split:
                        indexerLT += 1
                        if target == labels[0]:
                            indexer2LT += 1
                    elif float(inputs[n, i]) >= split:
                        indexerGT += 1
                        if target == labels[0]:
                            indexer2GT += 1
                countsLT.append(indexerLT)
                countsGT.append(indexerGT)
                partitionedCountsLT.append(indexer2LT)
                partitionedCountsGT.append(indexer2GT)
                indexerLT = 0
                indexerGT = 0
                indexer2LT = 0
                indexer2GT = 0

                if(len(targets) == 0):
                    product += 0
                elif (countsLT[0] - partitionedCountsLT[0] == 0 or partitionedCountsLT[0] == 0):
                    product += 0
                elif (countsGT[0] - partitionedCountsGT[0] == 0 or partitionedCountsGT[0] == 0):
                    product += 0
                else:
                    product += ((Decimal(((countsLT[0]/len(targets)) * (-1 * ((partitionedCountsLT[0] / countsLT[0]) * np.log2(partitionedCountsLT[0] / countsLT[0]) + (((countsLT[0] - partitionedCountsLT[0]) / countsLT[0]) * np.log2((countsLT[0] - partitionedCountsLT[0]) / countsLT[0])))))))
                             + (Decimal(((countsGT[0]/len(targets)) * (-1 * ((partitionedCountsGT[0] / countsGT[0]) * np.log2(partitionedCountsGT[0] / countsGT[0]) + (((countsGT[0] - partitionedCountsGT[0]) / countsGT[0]) * np.log2((countsGT[0] - partitionedCountsGT[0]) / countsGT[0]))))))))

                currentContInfoGain = infod - product
                if(currentContInfoGain > bestContInfoGain):
                    bestContInfoGain = currentContInfoGain
                    bestContSplitAttribute = attribute[0]
                    bestContSplitAttributeIndex = i
                    bestContSplit = split
        else:
            # The math that actually calculates the information gain
            for l, count in enumerate(counts):
                if(len(targets) == 0):
                    product += 0
                elif (count - targetedCounts[l] == 0 or targetedCounts[l] == 0):
                    product += 0
                else:
                    product = (product + Decimal(((count/len(targets)) * (-1 * ((targetedCounts[l] / count) * np.log2(targetedCounts[l] / count) + (((count - targetedCounts[l]) / count) * np.log2((count - targetedCounts[l]) / count)))))))
            
            currentInfoGain = infod - product
            if(currentInfoGain > bestInfoGain):
                bestInfoGain = currentInfoGain
                bestSplitAttribute = attribute[0]
                bestSplitAttributeIndex = i

        if bestContInfoGain > bestInfoGain:
            bestInfoGain = bestContInfoGain
            bestSplitAttribute = bestContSplitAttribute
            bestSplitAttributeIndex = bestContSplitAttributeIndex
            # usedSplits.append(bestContSplitAttribute + str(bestContSplit))
        else:
            continuous = False
        
        product = 0
        targetedCounts.clear()
        counts.clear()

    return bestSplitAttribute, bestSplitAttributeIndex, continuous, usedSplit, bestContSplit

def splitOnAttribute(splitAttribute, bestSplitIndex, targets, inputs, attributes, usedSplits):
    splitInput = []
    splitTarget = []
    splitInputList = []
    splitTargetList = []
    splitAttributeIndex = bestSplitIndex

    # for attribute in attributes:
    #     if attribute[0] == splitAttribute:
    #         splitAttributeIndex = attributes.index(attribute)
    #         break
    for j, value in enumerate(attributes[splitAttributeIndex]):
        if j == 0:
            continue
        for i, input in enumerate(inputs):


            # # if attribute[1] == attribute[2]: we are dealing with a continuous attribute
            # if attributes[splitAttributeIndex][1] == attributes[splitAttributeIndex][2]:
            #     # the first index indicates this is the less than or equal to split
            #     if j == 1:
            #         if float(input[splitAttributeIndex]) <= float(attributes[splitAttributeIndex][1]):
            #             splitInput.append(input)
            #             splitTarget.append(targets[i])
            #     # the second index indicates this is the greater than split
            #     elif j == 2:
            #         if float(input[splitAttributeIndex]) > float(attributes[splitAttributeIndex][1]):
            #             splitInput.append(input)
            #             splitTarget.append(targets[i])


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

def splitOnContinuousAttribute(splitAttribute, bestSplitIndex, targets, inputs, attributes, usedSplits, bestContSplitValue):
    splitInput = []
    splitTarget = []
    splitInputList = []
    splitTargetList = []
    splitAttributeIndex = bestSplitIndex

    for j in range (3):
        if j == 0:
            continue
        for i, input in enumerate(inputs):
            if j == 1:
                if float(input[splitAttributeIndex]) <= bestContSplitValue:
                    splitInput.append(input)
                    splitTarget.append(targets[i])
            elif j == 2:
                if float(input[splitAttributeIndex]) > bestContSplitValue:
                    splitInput.append(input)
                    splitTarget.append(targets[i])
        splitInputList.append(np.array(splitInput))
        splitTargetList.append(np.array(splitTarget))
        splitInput.clear()
        splitTarget.clear()
    
    newAttributes = attributes.copy()
        
    return splitInputList, splitTargetList, newAttributes

def generateDecisionTree(targets, inputs, attributes, labels, usedSplits, depth=0):
    
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

    bestSplit, bestSplitIndex, isContinuous, newUsedSplit, bestContSplitValue = infoAttributeD(targets, inputs, attributes, labels, usedSplits)
    if isContinuous:
        newInputs, newTargets, newAttributes = splitOnContinuousAttribute(bestSplit, bestSplitIndex, targets, inputs, attributes, newUsedSplit, bestContSplitValue)
    else:
        newInputs, newTargets, newAttributes = splitOnAttribute(bestSplit, bestSplitIndex, targets, inputs, attributes, usedSplits)

    for i, input in enumerate(newInputs):
        if isContinuous:
            if i == 0:
                print("\t" * (depth) + bestSplit + " <= " + str(round(bestContSplitValue, 2)) + ":")
            else:
                print("\t" * (depth) + bestSplit + " > " + str(round(bestContSplitValue, 2)) + ":")
        else:
            print("\t" * (depth) + bestSplit + " = " + attributes[bestSplitIndex][i+1] + ":")
        usedSplits.append(newUsedSplit)
        generateDecisionTree(newTargets[i], input, newAttributes, labels,  usedSplits, depth+1)



if __name__ == "__main__":
    main()