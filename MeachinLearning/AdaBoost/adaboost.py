## AdaBoost，二分类问题
## 弱分类器选择：单层决策树（仅基于单个特征来做决策，由于这颗树有一次分裂过程因此实际上就是一个树桩）
import numpy as np
import sys
sys.path.append('.')
from boost import *
# from   import *
def loadSimpData():
    dataMat = np.array([[1., 2.1],
    [2., 1.1],
    [1.3, 1.],
    [1., 1.],
    [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat,classLabels

D = np.array(np.ones([5, 1])/5)
dataMat, classLabels = loadSimpData()
boost.buildStump(dataMat, classLabels, D)