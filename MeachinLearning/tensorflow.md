# tensorflow functions
## tf.Session
```py
    sess = tf.Session()
    sess.run(xx)
    sess.close()

    with tf.Session() as sess:
        sess.run()
```

## tf.Variable

在 Tensorflow 中，定义了某字符串是变量，它才是变量，这一点是与 Python 所不同的。
定义语法： state = tf.Variable()

```py

    import tensorflow as tf

    state = tf.Variable(0, name='counter')

    # 定义常量 one
    one = tf.constant(1)

    # 定义加法步骤 (注: 此步并没有直接计算)
    new_value = tf.add(state, one)

    # 将 State 更新成 new_value
    update = tf.assign(state, new_value)
```

如果在 Tensorflow 中设定了变量，那么初始化变量是最重要的！！所以定义了变量以后, 一定要定义 init = tf.initialize_all_variables() .
到这里变量还是没有被激活，需要再在 sess 里, sess.run(init) , 激活 init 这一步.

```py
    # 如果定义 Variable, 就一定要 initialize
    # init = tf.initialize_all_variables() # tf 马上就要废弃这种写法
    init = tf.global_variables_initializer()  # 替换成这样就好
    
    # 使用 Session
    with tf.Session() as sess:
        sess.run(init)
        for _ in range(3):
            sess.run(update)
            print(sess.run(state)) 
            # 注意：直接 print(state) 不起作用！！一定要把 sess 的指针指向 state 再进行 print 才能得到想要的结果！
```

**tf.Variable(initializer, name)**

- initializer:初始化参数
- name: 可自定义变量名称
- eg: v1 = tf.Variable(tf.ones([4,3]), name = 'v1')

```py
tf.Variable.init(init_value, trainable = True, collections = None, validate_shape = True, name = None)
```

| 参数名称 | 参数类型 | 含 义|
|------|------|------|
|init_value| 所有可以转换为tensor 的类型|变量的初始值|
|trainable|bool|如果为True,会把它加入到GraphKeys.TRAINABLE_VARIABLES,才能对它使用Optimizer|
|collections|list|制定该图变量的类型，默认为[GraphKeys.GLOBAL_VARIABLES]|
|validate_shape|bool|如果为False,则不进行类型和维度检查|
|name|string|变量的名称，如果没有指定则系统会自动分配一个唯一的值|

```py
    v = tf.Variable(3,name='v')
    v2 = v.assign(5)
    sess = tf.InteractiveSession()
    sess.run(v.initializer)
    sess.run(v) -->3
    sess.run(v2) -->5
```

## tf.placeholder

**placeholder** 是tensorflow中的占位符，暂时存储变量。
Tensorflow 如果想要从外部传入data, 那就需要用到 tf.placeholder(), 然后以这种形式传输数据 

```py
sess.run(***, feed_dict={input: **}).
```

如下：

```py
    import tensorflow as tf

    #在 Tensorflow 中需要定义 placeholder 的 type ，一般为 float32 形式
    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)

    # mul = multiply 是将input1和input2 做乘法运算，并输出为 output 
    ouput = tf.multiply(input1, input2)
```

接下来, 传值的工作交给了 sess.run() , 需要传入的值放在了feed_dict={} 并一一对应每一个 input. placeholder 与 feed_dict={} 是绑定在一起出现的。

```py
    with tf.Session() as sess:
    print(sess.run(ouput, feed_dict={input1: [7.], input2: [2.]}))
    # [ 14.]
```

## 激励函数

**添加layer**
定义add_layer()
在 Tensorflow 里定义一个添加层的函数可以很容易的添加神经层,为之后的添加省下不少时间.神经层里常见的参数通常有weights、biases和激励函数。

首先，我们需要导入tensorflow模块。
然后定义添加神经层的函数def add_layer(),它有四个参数：输入值、输入的大小、输出的大小和激励函数，我们设定默认的激励函数是None。

```py
    def add_layer(inputs, in_size, out_size, activation_function=None):
```

接下来，我们开始定义weights和biases。
因为在生成初始参数时，随机变量(normal distribution)会比全部为0要好很多，所以我们这里的weights为一个in_size行, out_size列的随机变量矩阵。

```py
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
```

在机器学习中，biases的推荐值不为0，所以我们这里是在0向量的基础上又加了0.1。

```py
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
```

下面，我们定义Wx_plus_b, 即神经网络未激活的值。其中，tf.matmul()是矩阵的乘法。

```py
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
```

当activation_function——激励函数为None时，输出就是当前的预测值——Wx_plus_b，不为None时，就把Wx_plus_b传到activation_function()函数中得到输出。

```py
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
```

最后，返回输出，添加一个神经层的函数——def add_layer()就定义好了。

```py
    return outputs
```

## tf.nn.conv2d 卷积函数

```py
    tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
```

- input: 卷积输入。Tensor(tf.Constant, tf.Variable, tf.placeholder), [batch, in_height, in_width, in_channel]
- filter: 卷积核。Tensor[filter_height, filter_width, in_channel, out_channel]
- strides: 卷积核在各个维度移动的步长。一个长度为4的一维向量list [stride_batch, stride_height, stride_width, stride_channel]
- padding: 对输入input的填充方法。'SAME' 和 'VALID' 两种.   SAME 表示卷积核可以停留在图像边缘。

```py
##eg: 输入1个通道的5*5图像,卷积核为1个通道3*3, 步长 1. 输出1个通道的3*3  feature map

    input = tf.Variable(tf.random_normal([1,5,5,1]))
    filter = tf.Variable(tf.random_normal([3,3,1,1]))
    op = tf.nn.conv2d(input, filter, strides=[1,1,1,1], padding='VALID')

    #通道为5时， 输出为5个通道的 3*3 feature map

    input = tf.Variable(tf.random_normal([1,5,5,5]))
    filter = tf.Variable(tf.random_normal([3,3,5,1]))
    op = tf.nn.conv2d(input, filter, strides=[1,1,1,1], padding='VALID')

    #当padding='SAME'时，运行在边缘停留，输出为5*5的feature map.

    input = tf.Variable(tf.random_normal([1,5,5,5]))
    filter = tf.Variable(tf.random_normal([3,3,5,1]))
    op = tf.nn.conv2d(input, filter, strides=[1,1,1,1], padding='SAME')

    #当有多个卷积核时，输出多个feature map.如下输出7个5*5 feature map.

    input = tf.Variable(tf.random_normal([1,5,5,5]))
    filter = tf.Variable(tf.random_normal([3,3,5,7]))
    op = tf.nn.conv2d(input, filter, strides=[1,1,1,1], padding='SAME')

    #当步长不为1 时，因为图片只有两维，通常strides = [1, stride, stride, 1]

    input = tf.Variable(tf.random_normal([1,5,5,5]))
    filter = tf.Variable(tf.random_normal([3,3,5,7]))
    op = tf.nn.conv2d(input, filter, strides=[1,2,2,1], padding='SAME')

    #如果batch 值不为1， 同时输入10张图片，每张图片都有7张 5*5 feature map,输出shape 就是[10,5,5,7]

    input = tf.Variable(tf.random_normal([10,5,5,5]))
    filter = tf.Variable(tf.random_normal([3,3,5,7]))
    op = tf.nn.conv2d(input, filter, strides=[1,1,1,1], padding='SAME')
```

## tf.nn.max_pool 池化函数

```py
    tf.nn.max_pool(value, ksize, strides, padding, name=None)
```

  - value: 池化输入，一般池化层接在卷积层的后面，所以输入通常是feature map [bath, in_height, in_width, in_channels]
  - ksize: 池化窗口大小，四维向量[batch, height, width, channel]. 一般取值为 [1, height, width, 1] 因为不想在 batch 和 channel上池化。
  - strides: 步长，四维向量[bath, height, width, channel]，一般取值为[1, height, width,1]
  - padding: 填充方式。'SAME','VALID'


## tf.truncated_normal 

**从截断的正态分布中输出随机值**

```py
    tf.truncated_normal(shape, mean=0.0, stddev=1.0,dtype=tf.float32, seed=None, name=None)
```

- shape: 生成张量的维度
- mean: 均值
- stddev: 方差
- seed: 一个整数，当设置之后每次生成的随机数都一样
- name: 操作的名字
在truncated_normal中如果取值在区间（mean-2*stddev, mean+2*stddev）之外则重新选择，保证生成的值都在均值附近。

```py
    生成的值服从具有指定均值和方差的正态分布，如果生成值大于平均值2个标准差  则丢弃重新生成。
    在正态分布曲线中，横轴区间（mean-stddev, mean+stddev）内的面积为68.268949%
    横轴区间（mean-2*stddev, mean+2*stddev）内的面积为95449974%
    横轴区间（mean-3*stddev, mean+3*stddev）内的面积为99.730030%
    落在3*stddev 以外的概率小于千分之三，基本不可能发生，基本上可以把区间(mean-3*stddev, mean+3*stddev)看做随机变量x实际可能的取值区间，这称为正态分布的3*stddev 原则。
```

## tf.random_normal

```py
    tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
```

从正态分布中输出随机值。

## tf.name_scope 和 tf.variable_scope
结合tensorflow 创建两种方式 tf.get_variable(), tf.Variable()一起说明
在tf.name_scope下：

- tf.get_variable()创建的变量名不受tf.name_scope的影响，即创建的变量的name没有name_scope定义的前缀，而且在未指定共享变量时，如果重名会报错。
  
```py
with tf.name_scope('name_scope_1'):
    var1 = tf.get_variable(name='var1, shape=[1],dtype=tf.float32)
    var2 = tf.get_variable(name='var1', shape=[1],dtype=tf.float32)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(var1.name, sess.run(var1))
    print(var2.name, sess.run(var2))
#ValueError variable var1 already exists, disallowed.
```
- tf.Variable() 会自动检测有没有变量重名，如果有会自行处理

```py
with tf.name_scope('name_scope_x):
    var1 = tf.get_variable(name='var1', shape=[1],dtype=tf.float32)
    var3=tf.Variable(name='var2',initial_value=[2],dtype=tf.float32)
    var4=tf.Variable(name='var2',initial_value=[2],dtype=tf.float32)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(var1.name, sess.run(var1))
    print(var3.name,sess.run(var3))
    print(var4.name, sess.run(var4))
#输出结果
#var1:0 [-0.30036557] 前面不含指定的name_scope_x
#name_scope_x/var2:0 [2.]
#name_scope_x/var2_1:0[2.] 可以看到变量名自行变成 var2_1 避免冲突。
```
要共享变量需要用 tf.variable_scope()

对于使用tf.Variable来说，tf.name_scope和tf.variable_scope功能一样，都是给变量加前缀，相当于分类管理，模块化。
对于tf.get_variable来说，tf.name_scope对其无效，也就是说tf认为当你使用tf.get_variable时，你只归属于tf.variable_scope来管理共享与否。

```py
with tf.name_scope('name_sp1') as scp1:
    with tf.variable_scope('var_scp2') as scp2:
        with tf.name_scope('name_scp3') as scp3:
            a = tf.Variable('a')
            b = tf.get_variable('b')
##等价于
with tf.name_scope('name_sp1') as scp1:
    with tf.name_scope('name_sp2') as scp2:
        with tf.name_scope('name_scp3') as scp3:
            a = tf.Variable('a')

with tf.variable_scope('var_scp2') as scp2:
        b = tf.get_variable('b')
```

## tf.expand_dims
维度增加一维

```py
tf.expand_dims(inputs, dim, name=None)
```

当然，我们常用tf.reshape(input, shape=[])也可以达到相同效果，但是有些时候在构建图的过程中，placeholder没有被feed具体的值，这时就会包下面的错误：TypeError: Expected binary or unicode string, got 1 

```py
# 't' is a tensor of shape [2]
shape(expand_dims(t, 0)) ==> [1, 2]
shape(expand_dims(t, 1)) ==> [2, 1]
shape(expand_dims(t, -1)) ==> [2, 1]

# 't2' is a tensor of shape [2, 3, 5]
shape(expand_dims(t2, 0)) ==> [1, 2, 3, 5]
shape(expand_dims(t2, 2)) ==> [2, 3, 1, 5]
shape(expand_dims(t2, 3)) ==> [2, 3, 5, 1]
```

## tf.range

创建数字序列

```py
    tf.range(limit, delta=1, dtype=None, name='range')
    tf.range(start, limit, delta=1, dtype=None, name='range')
```

创建一个数字序列,该数字开始于 start 并且将 delta 增量扩展到不包括 limit 的序列.
除非明确提供,否则得到的张量的 dtype 是从输入中推断出来的.像 Python 内置的 range,start 默认为 0,所以 range(n) = range(0, n).

## tf.concat

将张量沿一个维度串联

```py
tf.concat(values, axis, name='concat')
```

## tf.pack/tf.stack

将 values 里面的张量 打包成一个张量

```py
    tf.stack(values, name='pack')
    ##eg:
    a = tf.constant([1,2,3])  
    b=tf.constant([4,5,6])sess.run([a,b]) 
    ##output:
    ## [array([1, 2, 3]), array([4, 5, 6])]
    sess.run(tf.stack([a,b],name='rank'))  
    ##output：[[1,2,3],[4,5,6]]
```

 
