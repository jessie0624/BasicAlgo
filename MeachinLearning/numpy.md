
## numpy中的multiply 和* 和 matul 的区别

1、对于矩阵（matrix）而言，multiply是对应元素相乘，而 *  、np.matmul() 函数 与 np.dot()函数 相当于矩阵乘法（矢量积），对应的列数和行数必须满足乘法规则；如果希望以数量积的方式进行，则必须使用 np.multiply 函数，如下所示：

a = np.mat([[1, 2, 3, 4, 5]])
b = np.mat([[1,2,3,4,5]])
c=np.multiply(a,b)
print(c)
结果是[[ 1  4  9 16 25]]
a = np.mat([[1, 2, 3, 4, 5]])
b = np.mat([ [1],[2],[3],[4],[5] ] )
d=a*b
print(d)   #a是shape（1,5），b是shape（5,1），结果是一个实数
结果是[[55]]

2、对于数组（Array）而言，* 与 multiply均表示的是数量积（即对应元素的乘积相加），np.matmul与np.dot表示的是矢量积（即矩阵乘法）。

## python中转置的优先级高于乘法运算 

例如：
a = np.mat([[2, 3, 4]])
b = np.mat([[1,2,3]] )
d=a*b.T
print(d)
结果是 [[20]]
其中a为1行3列，b也为1行3列，按理来说直接计算a*b是不能运算，但是计算d=a*b.T是可以的，结果是20，说明运算顺序是先转置再计算a与b转置的积，*作为矩阵乘法，值得注意的在执行*运算的时候必须符合行列原则。

## numpy中tile（）函数的用法

b = tile(a,(m,n)):即是把a数组里面的元素复制n次放进一个数组c中，然后再把数组c复制m次放进一个数组b中，通俗地讲就是将a在行方向上复制m次，在列方向上复制n次。

python中的 sum 和 np.sum 是不一样的，如果只写sum的话，表示的是数组中对应的维度相加，如果写 np.sum 的话，表示一个数组中的维数和列数上的数都加在一起

```py
data1=mat(zeros((3,3)));
#创建一个3*3的零矩阵，矩阵这里zeros函数的参数是一个tuple类型(3,3)
data2=mat(ones((2,4)));
#创建一个2*4的1矩阵，默认是浮点型的数据，如果需要时int类型，可以使用dtype=int
data3=mat(random.rand(2,2));
#这里的random模块使用的是numpy中的random模块，random.rand(2,2)创建的是一个二维数组，需要将其转换成#matrix
data4=mat(random.randint(10,size=(3,3)));
#生成一个3*3的0-10之间的随机整数矩阵，如果需要指定下界则可以多加一个参数
data5=mat(random.randint(2,8,size=(2,5));
#产生一个2-8之间的随机整数矩阵
data6=mat(eye(2,2,dtype=int));
#产生一个2*2的对角矩阵
a1=[1,2,3];
a2=mat(diag(a1));
#生成一个对角线为1、2、3的对角矩阵
## https://blog.csdn.net/qq_30638831/article/details/79907684
```
