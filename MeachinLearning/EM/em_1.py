import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances_argmin

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':
    style = 'myself'
    np.random.seed(0) ##使得随机数据可预测 当设置相同的seed,每次生成的随机数据相同，如果不设置seed,每次生成的随机数据不同。
    mu1_fact = (0,0,0)
    cov_fact = np.identity(3) ## 3维单元矩阵
    data1 = np.random.multivariate_normal(mu1_fact, cov_fact, 400) ## 根据均值mu1_fact 和方差cov_fact 生成多元高斯分布 400组数据。
    mu2_fact = (2,2,1)
    cov_fact = np.identity(3)
    data2 = np.random.multivariate_normal(mu2_fact, cov_fact, 100)
    data = np.vstack((data1, data2))
    y = np.array([True] * 400 + [False] * 100)

    if style == 'sklearn':
        g = GaussianMixture(n_components=2, covariance_type='full', tol=1e-6, max_iter=1000)
        '''
        GaussianMixture(n_components=1, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1, init_params='kmeans',
        weights_init=None, means_init=None, precisions_init=None, random_state=None, warm_start=False, verbose=0, verbose_interval=10)
        高斯混合模型
        代表高斯混合模型概率分布，可以用来估计高斯混合模型分布的参数。
        n_components: 混合的成分。（默认是1）本例中混合了2种高斯分布模型，所以值为2。
        covariance_type:{'full','tied','diag','spherical'} 协方差类型
            'full': 每一个高斯分布都有自己的方差矩阵.
            'tied': 所有的高斯分布共享同一个协方差矩阵.
            'diag': 每个高斯分布都有自己的对角协方差矩阵.
            'spherical':每个高斯分布都有自己的单方差.
        tol: 当lower bound 平均增益低于 这个阈值时，EM 迭代将会停止
        reg_covar:非负正则 添加到对角协方差矩阵
        warm_start : bool, default to False.
            If 'warm_start' is True, the solution of the last fitting is used as
            initialization for the next call of fit(). This can speed up
            convergence when fit is called several time on similar problems.
        
        ###Attributes:
        weights_: array-like, shape(n_components,) 每一个混合高斯的权重
        means_: array-like,shape(n_components, n_features) 每一个混合高斯的均值
        covariances_: array-like 每一个混合高斯的协方差.shape 和 covariance_type 有关 
        precisions_: array-like 准确度矩阵。
            The precision matrices for each component in the mixture. A precision
            matrix is the inverse of a covariance matrix. A covariance matrix is
            symmetric positive definite so the mixture of Gaussian can be
            equivalently parameterized by the precision matrices. Storing the
            precision matrices instead of the covariance matrices makes it more
            efficient to compute the log-likelihood of new samples at test time.
        precisions_cholesky_ : array-like
            The cholesky decomposition of the precision matrices of each mixture
            component. A precision matrix is the inverse of a covariance matrix.
            A covariance matrix is symmetric positive definite so the mixture of
            Gaussian can be equivalently parameterized by the precision matrices.
            Storing the precision matrices instead of the covariance matrices makes
            it more efficient to compute the log-likelihood of new samples at test
            time.

        converged_ : bool
            True when convergence was reached in fit(), False otherwise.
        n_iter_ : int
            Number of step used by the best fit of EM to reach the convergence.
        lower_bound_ : float
            Log-likelihood of the best fit of EM.

        '''

        g.fit(data)
        print('pro:{}'.format(g.weights_[0]))
        print('mean:{}'.format(g.means_))
        print('std:{}'.format(g.covariances_))
        mu1, mu2 = g.means_
        sigma1, sigma2 = g.covariances_
    else:
        num_iter = 100
        n, d = data.shape
        mu1 = data.min(axis=0)## 随机指定均值1 设最小值是mu1,最大值是mu2 
        mu2 = data.max(axis=0)## 随机指定均值2
        sigma1 = np.identity(d)
        sigma2 = np.identity(d)
        pi = 0.5
        ##EM:
        for i in range(num_iter):
            ##E-step
            norm1 = multivariate_normal(mu1, sigma1)
            norm2 = multivariate_normal(mu2, sigma2)
            tau1 = pi* norm1.pdf(data)
            tau2 = (1-pi)* norm2.pdf(data)
            gamma = tau1 / (tau1 + tau2) ## 由第一个分组组成的概率gamma, 由第二个分组组成的概率1-gamma

            ##M-step
            mu1 = np.dot(gamma, data) /np.sum(gamma)
            mu2 = np.dot((1-gamma), data)/np.sum(1-gamma)
            sigma1 = np.dot(gamma * (data - mu1).T, data - mu1) / np.sum(gamma)
            sigma2 = np.dot((1-gamma) * (data - mu2).T, data - mu2)/ np.sum(1-gamma)
            pi = np.sum(gamma)/n
            print(i, mu1, mu2)
        print('\n 类别概率：{}'.format(pi))
        print('\n 均值：mu1 {}  mu2 {}'.format(mu1,mu2))
        print('\n方差: sigma1 {}  sigma2{}'.format(sigma1, sigma2))


    norm1 = multivariate_normal(mu1, sigma1) ### multivariate() 多维正态分布/高斯分布(均值，方差) 协方差矩阵必须是正定矩阵(需要转置和逆矩阵)
    norm2 = multivariate_normal(mu2, sigma2)
    tau1 = norm1.pdf(data) ## 基于norm1 生成data的概率密度函数 即500个概率密度值  
    tau2 = norm2.pdf(data)## 基于norm2 生成data的概率密度函数 即500个概率密度值  
    ##如何区分tau1 和tau2? tau1>tau2 属于一个类别，不大于属于另外一个类别
    '''
    multivariate_normal() 多维高斯分布
    method：
        pdf(x, mean=None, cov=1, allow_singular=False) probability density fucntion. 概率密度函数
        logpdf(x, mean=None, cov=1, allow_singular=False) log of probability density function.对数概率密度函数
        cdf(x, mean=None, cov=1, allow_singular=Flase, maxpts=1000000*dim, abseps=1e-5) Cumulative distribution function 累积分布
        logcdf(x, mean=None, cov=1, allow_singular=False, maxpts=1000000*dim, abseps=1e-5, releps=1e-5) 对数累积分布
        rvs(mean=None, cov=1, size=1, random_state=None) Draw random samples from a multivariate normal distribution. 绘图
        entropy() Compute the differential entropy of the multivariate normal.计算熵
    '''

    fig = plt.figure(figsize=(13,7), facecolor='w')
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(data[:,0], data[:,1], data[:,2], c='b', s=30, marker='o', depthshade=True)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(u'original data', fontsize=18)
    ax = fig.add_subplot(122, projection='3d')
    order = pairwise_distances_argmin([mu1_fact, mu2_fact], [mu1,mu2], metric='euclidean')##用来区分 我们算的mu1,mu2 和实际的mu1_fact mu2_fact 是否一致 或者是否相反。
    ### 返回order [0,1] 或者[1,0] 如果order[0]=0 那么mu1 离 mu1_fact 近，mu2 距离mu2_fact 近。 
    #### pairwise_distances_argmin(X,Y)  return Y[argmin[i], :] is the row in Y that is closest to X[i, :]. 返回Y 中离x最近最近的顺序。
    
    print('order:{}'.format(order))
    if order[0] == 0:
        c1 = tau1 > tau2
    else:
        c1 = tau1 < tau2
    c2 = ~c1
    acc = np.mean(y==c1)
    print(u'accuracy %.2f%%' % (100*acc))
    ax.scatter(data[c1, 0], data[c1, 1], data[c1, 2], c='r', s=30, marker='o', depthshade=True)
    ax.scatter(data[c2, 0], data[c2, 1], data[c2, 2], c='g', s=30, marker='^', depthshade=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(u'EM classification', fontsize=18)
    # # plt.suptitle(u'EM算法的实现', fontsize=20)
    # # plt.subplots_adjust(top=0.92)
    plt.tight_layout()
    plt.show()
        
        

