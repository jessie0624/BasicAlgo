class node:
    def __init__(self, fea=-1, value=None, results=None, right=None, left=None):
        self.fea = fea
        self.value = value
        self.results = results
        self.right = right
        self.left = left

def label_unique_cnt(data):
    labels = [examples[-1] for examples in data]
    unique_labels = set(labels)
    label_count = {}
    for value in unique_labels:
        if value not in label_count:
            label_count[value] = 0
        label_count[value] += 1
    return label_count

def cal_gini_index(data):
    
    numberData = len(data)
    if not  numberData:
        return 0
    label_counts = label_unique_cnt(data)

    gini = 0
    for key  in label_counts:
        gini += pow(label_counts[key],2)
    gini = 1 - float(gini) / pow(numberData, 2)
    return gini

def splitData(data, fea, value):
    # fea: best feature
    # value: best value
    left = []
    right = []
    for x in data:
        if x[fea] >= value:
            right.append(x)
        else:
            left.append(x)
    return (right, left)

def buildTree(data):
    if len(data) ==0 :
        return node()
    
    currentGini = cal_gini_index(data)
    bestGain = 0.0
    bestCriteria = None ##save best fea and best value
    bestSets = None ## save best dataset

    feature_num = len(data[0]) - 1
    for fea in range(feature_num):
        #feature_values = {}
        feature_values = [sample[fea] for sample in data]
        unique_values = set(feature_values)
        for value in unique_values:
            (set1, set2) = splitData(data, fea, value)
            newGini = float(len(set1) * cal_gini_index(set1) + len(set2) * cal_gini_index(set2))/len(data)
            gain = currentGini - newGini
            if gain > bestGain and len(set1) >0 and len(set2) >0:
                bestGain = gain
                bestCriteria = (fea,value)
                bestSets = (set1, set2)
    if bestGain > 0:
        right = buildTree(bestSets[0])
        left = buildTree(bestSets[1])
        return node(fea=bestCriteria[0], value=bestCriteria[1],right=right, left=left)
    else:
        return node(results=label_unique_cnt(data))

def predict(sample, tree):
    if tree.results !=None:
        return tree.results
    else:
        val_sample = sample[tree.fea]
        branch = None
        if val_sample >= tree.value:
            branch = tree.right
        else:
            branch = tree.left
        return predict(sample, branch)

           
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
a = buildTree(myDat)
print(a.fea)
print(a.left)
print(a.right)
print(a.value)

print(predict(createTestDataSet()[0], a))

       


