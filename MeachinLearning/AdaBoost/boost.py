import numpy as np

##stumpClassify
## 通过阈值比较进行分类，所有在阈值一边的数据会被分到类别-1，而在另外一边的数据分到类别+1
## 该函数通过数组过滤来实现，首先将返回数组全部元素设为1，然后将不满足不等式要求的元素设置为-1
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = np.ones([np.shape(dataMatrix)[0], 1])
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

## buildStump 遍历stumpClassify 函数所有可能输入值，并找到数据集上最佳的单层决策树。
## 这里的‘最佳’是基于数据的权重向量D来定义的。
def buildStump(dataArr, classLabels, D):
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m,n = np.shape(dataMatrix)  ##m 行，n列   m 个样本，n-1个特征。
    numSteps = 10.0
    bestStump = {} ## 最优树桩
    bestClasEst = np.mat(np.zeros([m, 1])) ## 最好的分类
    minError = np.inf ## 最小错误率
    for i in range(n): ## 对数据集的每一个特征循环
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin)/numSteps
        for j in range(-1, int(numSteps) + 1): ## 对每个步长
            for inequal in ['lt', 'gt']: ## 对每个不等号
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = np.mat(np.ones([m, 1]))
                #errArr[predictedVals == labelMat] = 0
                for k in range(len(errArr)):
                    if predictedVals[k,0] == labelMat[k,0]:
                        errArr[k,0] = 0
                weightedError = D.T.dot(errArr)
                print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f"
                %(i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst

## 基于单层决策树的AdaBoost训练过程
def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m =np.shape(dataArr)[0]
    D = np.mat(np.ones([m,1]))/m ## 初始化权重
    aggClassEst = np.mat(np.zeros([m,1]))

    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print('D:', D.T)
        alpha = float(0.5 * np.log((1.0 - error)/max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print('classEnst:', classEst.T)
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D/D.sum()
        aggClassEst += alpha * classEst
        print('aggClassEst:', aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones([m,1]))
        errorRate = aggErrors.sum()/m
        print('total erro:', errorRate, "\n")
        if errorRate == 0.0:
            break
    return weakClassArr, aggClassEst

## AdaBoost 分类函数
def adaClassify(datToClass, classifierArr):
    dataMatrix = np.mat(datToClass)
    m = np.shape(datToClass)[0]
    aggClassEst = np.mat(np.zeros([m,1]))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],
                                classifierArr[i]['thresh'],
                                classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
    return np.sign(aggClassEst)



def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0, 1.0)
    ySum = 0.0
    numPosClas = sum(np.array(classLabels) == 1)
    yStep = 1/float(numPosClas)
    xStep = 1/float(len(classLabels) - numPosClas)
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    ax = plt.subplot(1,1,1)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1: ##y 是正阳率(正例的比例)，初始化为1 说明TP/TP+NP =1 即NP=0
            delX = 0; delY = yStep
        else:
            delX = xStep; delY = 0
            ySum += cur[1]
        ax.plot([cur[0], cur[0]-delX], [cur[1] , cur[1] - delY], c='b')
        cur = (cur[0] - delX, cur[1] - delY)
    ax.plot([0,1], [0,1], 'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Position Rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print('the Area under the curve is :%f' % (ySum * xStep ))
##test

# def loadSimpData():
#     dataMat = np.mat([[1., 2.1],
#     [2., 1.1],
#     [1.3, 1.],
#     [1., 1.],
#     [2., 1.]])
#     classLabels = np.mat([[1.0, 1.0, -1.0, -1.0, 1.0]])
#     return dataMat,classLabels

# D = np.mat(np.ones([5, 1])/5)
# dataMat, classLabels = loadSimpData()
# # print(buildStump(dataMat, classLabels, D))

# classifierArr = adaBoostTrainDS(dataMat, classLabels, 9)
# #adaClassify(np.array([[0,0]]), classifierArr)
# adaClassify([[0,0],[5,5]], classifierArr)


## Horse-Colic predict
def loadDataSet(filename):
    f = open(filename)
    ret_arr = []
    label = []
    for line in f.readlines():
        newLine = [item for item in list(line.strip().split(' '))]
        newLine[22],newLine[-1] = newLine[-1],newLine[22] ## feature 23 represents horse survive,pick as label
        tmp = []
        if newLine[-1] != '?': ## drop the line without label
            for attr in newLine[:-1]:
                if attr == '?': ## fill 0 for missing feature
                    attr = 0
                tmp.append(float(attr))
            ret_arr.append(tmp)
            label.append(int(newLine[-1]) if int(newLine[-1]) == 1 else -1)
    return np.mat(ret_arr),label

import os
import os.path
dataArr,labelArr = loadDataSet(os.path.join(os.getcwd(),'horse-colic_data.txt'))
classifierArr, aggClassEst = adaBoostTrainDS(dataArr, labelArr, 20) ## 20 times 
plotROC(aggClassEst.T, labelArr)
testArr,testLabelArr = loadDataSet(os.path.join(os.getcwd(),'horse-colic_test.txt'))
prediciton10 = adaClassify(testArr,classifierArr) 
errArr = np.mat(np.ones((67,1))) ## 68 test data
print(errArr[prediciton10!=np.mat(testLabelArr).T].sum())




