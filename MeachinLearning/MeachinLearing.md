## 分类


### DecisionTree

熵：H(D) = -sum(p*log(p))

```py
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
```

- ID3:熵增益
    g(D,A) = g(D)-g(D|A)
- ID4.5: 熵增益率
    g(D,A) = (g(D)-g(D|A))/g(D|A)

```py
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
```

### CART

- gini 指数.
    gini(D) = 1-sum(p^2)

```py
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
```

### 集成学习

通过训练多个分类器，利用这些分类器来解决同一个问题，结合多个分类器对同一个问题的预测结果，给出最终预测结果。
集成学习泛化能力比单个学习算法强，在集成学习中，根据多个分类器学习方式不同，可以分为： Bagging 算法和 Boosting 算法。

- Bagging(Boostrap Aggregating)算法通过对训练样本有放回的抽取，由此产生多个训练数据子集，并在每一个训练子集上训练一个分类器，最终分类结果是由多个分类器的分类结果投票产生的。Bagging 算法的整个过程如下：
  
        原始训练样本--->通过boostrap 重新选择样本---> 分类器1，分类器2。。--> f(x) = 1/n*sum(F(i))

- Boosting算法通过顺序地给训练集中的数据项重新加权创造不同的基础学习器。核心思想是重复应用一个基础学习器来修改训练数据集，这样在预定数量的迭代下可以产生一系列的基础学习器。在训练开始，所有的数据项都被初始化为同一个权重，在初始化之后，每次增强的迭代都会生成一个适应加权之后的训练数据集的基础学习器。每一次迭代的错误率都会计算出来，而且正确划分的的数据项的权重会被降低，然后错误划分的数据项权重将会增大。Boosting 算法最终模型是一系列基础学习器的线性组合，而且系数依赖于各个基础学习器的表现。Boosting 有很多版本目前最广泛使用的是AdaBoost算法和GBDT算法。
  
        原始训练样本-->分类器1->加权->分类器2->加权->分类器n.  分类器1-n -->f(x) = sum(Wi * F(i))

    对于包含n个分类器的Boosting算法，依次利用训练样本对其进行学习，在每一个分类器中，其样本的权重是不一样的，如对第i+1个分类器来讲，第i个分类器会对每个样本进行评估，预测错误的样本，其权重会增加，反之会减小。训练好每一个分类器后，对每一个分类器的结果进行线性加权得到最终的预测结果。

## 随机森林




###