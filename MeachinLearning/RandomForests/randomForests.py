import numpy as np
from math import * 
import random as rd
from MeachinLearning.CART_classification import *

def choose_samples(data, k):
    '''
    从样本中随机选择样本及其特征
    input: data 原始数据集， k选择特征个数
    output: data_samples(list) 被选择出来的样本
            feature(list) 被选择的特征 index
    '''
    m,n = np.shape(data) ## 样本个数和特征个数
    ## 1，选择出K个特征的index
    feature = []
    for  j in  range(k):
        feature.append(rd.randint(0, n-2))  ## 第n-1列是label 
    ## 2, 选择出m个样本的index
    index = []
    for i in range(m):
        index.append(rd.randint(0, m-1))
    ## 3, 从data中选择出m个样本的k个特征，组成数据集data_samples
    data_samples = []
    for i in range(m):
        data_tmp = []
        for fea in feature:
            data_tmp.append(data[index[i]][fea])
        data_tmp.append(data[index[i]][-1])
        data_samples.append(data_tmp)
    return data_samples, feature



def random_forest_training(data_train, trees_num):
    '''
    input: data_train(list):训练数据
            trees_num:分类树的个数
    output: trees_result(list):每一颗树的最好划分
            trees_feature(list):每一颗树种对原始特征的选择
    '''
    trees_result = [] ##建立每一颗树的最好划分
    trees_feature = []
    n = np.shape(data_train)[1] ## 样本的维数,特征的个数
    if n >2:
        k = int(log(n-1,2)) + 1 ## 设置特征个数
    else:
        k = 1
    ##开始构建每一颗树
    for i in range(trees_num):
        ## 1,随机选择m个样本，k个特征
        data_sample, feature = choose_samples(data_train, k)
        ## 2, 构建每一棵树
        tree = buildTree(data_sample)
        ## 3, 保存训练好的分类树
        trees_result.append(tree)
        ## 4, 保存好该分类树使用到的特征
        trees_feature.append(feature)
    return trees_result, trees_feature



