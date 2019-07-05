# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

def expand(a, b):
    d = (b-a)*0.05
    return a-d, b+d

if __name__ == '__main__':
    data = np.loadtxt('18.HeightWeight.csv', dtype=np.float, delimiter=',', skiprows=1)
    print(data.shape)
    y, x = np.split(data, [1,], axis=1)
    
    x, x_test, y, y_test = train_test_split(x, y, train_size=0.6, random_state=0)
    
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
    x_min = np.min(x, axis=0)
    x_max = np.max(x, axis=0)
    gmm.fit(x)
    print('means: {}'.format(gmm.means_))
    print('covarviances:{}'.format(gmm.covariances_))
    y_hat = gmm.predict(x)
    y_test_hat = gmm.predict(x_test)
    change = (gmm.means_[0][0] > gmm.means_[1][0])
    if change:
        z = y_hat == 0
        y_hat[z] = 1
        y_hat[~z] = 0
        z = y_test_hat == 0
        y_test_hat[z] = 1
        y_test_hat[~z] = 0
    acc = np.mean(y_hat.ravel() == y.ravel()) ## 对应位置相等 则为1，不等则为0，均值就是 ：相等为1的总和/所有数据量  也就是预测的准确度。
    acc_test = np.mean(y_test_hat.ravel() == y_test.ravel())
    '''
    numpy 中的ravel(), flatten(), squeeze() 都是将多维数组转换为一维数组:
    ravel()：如果没有必要，不会产生源数据的副本。flatten(): 返回原数据的副本。squeeze()：只能对维数为1的维度降维。
    '''
    acc_str = u'训练集准确率：%.2f%%' % (acc * 100)
    acc_test_str = u'测试集准确率：%.2f%%' % (acc_test * 100)
    print(acc_str)
    print(acc_test_str)

    x1_min, x1_max = x[:,0].min(), x[:,0].max()
    x2_min, x2_max = x[:,1].min(), x[:,1].max()
    print('x1_min:{}, x1_max:{}'.format(x1_min, x1_max))
    print('x2_min:{}, x2_max:{}'.format(x2_min, x2_max))
    x1_min, x1_max = expand(x1_min, x2_max)
    x2_min, x2_max = expand(x2_min, x2_max)
    print('after expand : x1_min:{}, x1_max:{}'.format(x1_min, x1_max))
    print('after expand : x2_min:{}, x2_max:{}'.format(x2_min, x2_max))

    x1,x2 = np.mgrid[x1_min : x1_max:500j, x2_min : x2_max : 500j]
    grid_test = np.stack((x1.flat, x2.flat), axis=1)
    grid_hat = gmm.predict(grid_test)
    grid_hat = grid_hat.reshape(x1.shape)
    if change:
        z = grid_hat == 0
        grid_hat[z] = 1
        grid_hat[~z] = 0
    cm_light = mpl.colors.ListedColormap(['#FF8080', '#77E0A0'])
    cm_dark = mpl.colors.ListedColormap(['r', 'g'])
    plt.figure(figsize=(9,7), facecolor='w')
    plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)
    plt.scatter(x[:,0], x[:,1], s=50, c=y.ravel(), marker='o', cmap=cm_dark, edgecolors='k')
    plt.scatter(x_test[:,0], x_test[:,1], s=60, c=y_test.ravel(), marker='^', cmap=cm_dark, edgecolors='k')

    p = gmm.predict_proba(grid_test)
    p = p[:,0].reshape(x1.shape)
    CS = plt.contour(x1, x2, p, levels=(0.2, 0.5, 0.8), colors=list('rgb'), linewidths=2)
    plt.clabel(CS, fontsize=15, fmt='%.1f', inline=True)
    ax1_min, ax1_max, ax2_min, ax2_max = plt.axis()
    xx = 0.9 * ax1_min + 0.1 * ax1_max
    yy = 0.1 * ax2_min + 0.9 * ax2_max
    plt.text(xx, yy, acc_str, fontsize=18)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.xlabel(u'height(cm)', fontsize='large')
    plt.xlabel(u'weight(kg)', fontsize='large')
    plt.title(u'EM GMM parameter', fontsize=20)
    plt.grid()
    plt.show()
