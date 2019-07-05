import math
import matplotlib.pylab as plt  
import numpy as  np 
import codecs
import random

infinite = float(-2**31)

def log_normalize(a):
    s = 0
    for x in a:
        s += x
    if s == 0:
        print('Error..from log_normalize.')
        return 
    s = math.log(s)
    for i in range(len(a)):
        if  a[i] == 0:
            a[i] = infinite
        else:
            a[i] = math.log(a[i]) - s
 
 def log_sum(a):
    if not a:
        return infinite
    m = max(a)
    s = 0
    for t in a:
        s += math.exp(t-m)
    return m + math.log(s)

def calc_alpha(pi, A, B, o, alpha):  ##计算前向算法里面alpha[t][i] = (sum(alpha[t][j] * A[j][i])) * B[i][o[t]]
    for i in range(4): ### 状态取值有0,1,2,3,种情况。此处初始化 alpha[0][i]即在时刻0时 取状态i个概率。
        alpha[0][i] = pi[i] + B[i][ord(o[0])]
    T = len(o) ##观察序列的长度，即时刻T的总长度。
    temp = [0 for i in range(4)] ##
    del i 
    for t in range(1, T): ##从时刻1 到 T-1
        for i in range(4): ## 每个时刻可取的状态i 从0-3
            for j in range(4): ## 每个时刻的前一时刻可取的状态j 从0-3
                temp[j] = (alpha[t-1][j] + A[j][i]) ##
            alpha[t][i] = log_sum(temp)
            alpha[t][i] += B[i][ord(o[t])]

def calc_beta(pi, A, B, o, beta):
    T = len(o)
    for i in range(4):
        beta[T-1][i] = 1
    temp = [0 for i in range(4)]
    del i
    for t in range(T-2, -1, -1):
        for i in range(4):
            beta[t][i] = 0
            for j in range(4):
                temp[j] = A[i][j] + B[i][ord(o[t+1])] +beta[t+1][j]
            beta[t][i] += log_sum(temp)

def calc_gamma(alpha, beta, gamma):
    for t in range(len(alpha)):
        for i in range(4):
            gamma[t][i] = alpha[t][i] + beta[t][i]
        s = log_sum(gamma[t])
        for i in range(4):
            gamma[t][i] -= s

def calc_ksi(alpha, beta, A, B, o, ksi):
    T = len(o)
    temp = [0 for x in range(16)]
    for t in range(T-1):
        K=0
        for i in range(4):
            for j in range(4):
                ksi[t][i][j] = alpha[t][i] + A[i][j] + B[j][ord(o[t+1])] + beta[t+1][j]
                temp[k] = ksi[t][i][j]
                k += 1
        s = log_sum(temp)
        for i in range(4):
            for j in range(4):
                ksi[t][i][j] -= s

def bw(pi, A, B, alpha, beta, gamma, ksi, o):
    T = len(alpha)
    
