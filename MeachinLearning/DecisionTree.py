##DecisionTree
##ID3: g(D,A) = H(D)-H(D|A)
##CR4.5:g(D,A) = g(D,A)/H(D|A)
##CART: gini(t) for classification, minest sqrt(a-b)^2 for regression

##using ID3 creat tree
import operator
from math import log

def clacShannonEnt(dataset):
    numEntries = len(dataset)
    labelCounts = {}
    for featVec in dataset:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def splitDataSet(dataset, axis, value):
    # axis: feature to split dataset
    # value: the return dataset with axis = value
    # return: retDataSet without axist feature but axis == value
    retDataset = []
    for featVec in dataset:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:])
            retDataset.append(reduceFeatVec)
    return retDataset

def chooseBestFeatureToSplit(dataset):
    numFeatures = len(dataset[0]) - 1
    baseEntropy = clacShannonEnt(dataset)
    bestInfoGain =0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataset]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataset, i, value)
            prob = len(subDataSet)/float(len(dataset))
            newEntropy += prob * clacShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        #infoGain = (baseEntropy - newEntropy)/newEntropy  #ID4.5
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCount(classList):
    classCount = {}
    for key in classList:
        if key not in classCount:
            classCount[key] = 0
        classCount[key] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reversed= True)
    return sortedClassCount(0)

def createTree(dataset, labels):
    classList = [example[-1] for example in dataset]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataset[0]) == 1:
        return majorityCount(classList)
    bestFeature = chooseBestFeatureToSplit(dataset)
    bestFeatureLabel = labels[bestFeature]
    myTree = {bestFeatureLabel:{}}
    del labels[bestFeature]
    featValues = [example[bestFeature] for example in dataset]
    uniqueValues = set(featValues)
    for value in uniqueValues:
        subLabels = labels[:]
        myTree[bestFeatureLabel][value] = createTree(splitDataSet(dataset, bestFeature, value), subLabels)
    return myTree

def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    classLabel = None
    for key in list(secondDict.keys()):
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ =='dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel
            
###TEST

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels
def createTestDataSet():
    dataSet = [1, 1]
    labels = ['no surfacing', 'flippers']
    return dataSet,labels

myDat,labels = createDataSet()
myDat.append([0,0,'maybe'])
print(myDat)
mytree = createTree(myDat, labels)
print(mytree)
testDat, testFeature = createTestDataSet()
classifyLable = classify(mytree, testFeature, testDat )
print(classifyLable)
