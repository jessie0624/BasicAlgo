# tensorflow functions
## tf.Session

    sess = tf.Session()
    sess.run(xx)
    sess.close()

    with tf.Session() as sess:
        sess.run()

## tf.Variable
在 Tensorflow 中，定义了某字符串是变量，它才是变量，这一点是与 Python 所不同的。
定义语法： state = tf.Variable()

    import tensorflow as tf

    state = tf.Variable(0, name='counter')

    # 定义常量 one
    one = tf.constant(1)

    # 定义加法步骤 (注: 此步并没有直接计算)
    new_value = tf.add(state, one)

    # 将 State 更新成 new_value
    update = tf.assign(state, new_value)

如果在 Tensorflow 中设定了变量，那么初始化变量是最重要的！！所以定义了变量以后, 一定要定义 init = tf.initialize_all_variables() .
到这里变量还是没有被激活，需要再在 sess 里, sess.run(init) , 激活 init 这一步.
    
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

tf.Variable(initializer, name)
        initializer:初始化参数， name: 可自定义变量名称
        eg: v1 = tf.Variable(tf.ones([4,3]), name = 'v1')
tf.Variable.init(init_value, trainable = True, collections = None, validate_shape = True, name = None)
| 参数名称 | 参数类型 | 含 义|
|------|------|------|
|init_value| 所有可以转换为tensor 的类型|变量的初始值|
|trainable|bool|如果为True,会把它加入到GraphKeys.TRAINABLE_VARIABLES,才能对它使用Optimizer|
|collections|list|制定该图变量的类型，默认为[GraphKeys.GLOBAL_VARIABLES]|
|validate_shape|bool|如果为False,则不进行类型和维度检查|
|name|string|变量的名称，如果没有指定则系统会自动分配一个唯一的值|
        eg: v = tf.Variable(3,name='v')
            v2 = v.assign(5)
            sess = tf.InteractiveSession()
            sess.run(v.initializer)
            sess.run(v) -->3
            sess.run(v2) -->5

## tf.placeholder
placeholder 是tensorflow中的占位符，暂时存储变量。
Tensorflow 如果想要从外部传入data, 那就需要用到 tf.placeholder(), 然后以这种形式传输数据 sess.run(***, feed_dict={input: **}).
如下：
    
    import tensorflow as tf

    #在 Tensorflow 中需要定义 placeholder 的 type ，一般为 float32 形式
    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)

    # mul = multiply 是将input1和input2 做乘法运算，并输出为 output 
    ouput = tf.multiply(input1, input2)

接下来, 传值的工作交给了 sess.run() , 需要传入的值放在了feed_dict={} 并一一对应每一个 input. placeholder 与 feed_dict={} 是绑定在一起出现的。
    
    with tf.Session() as sess:
    print(sess.run(ouput, feed_dict={input1: [7.], input2: [2.]}))
    # [ 14.]

## 激励函数

## 添加layer
定义add_layer()
在 Tensorflow 里定义一个添加层的函数可以很容易的添加神经层,为之后的添加省下不少时间.
神经层里常见的参数通常有weights、biases和激励函数。

首先，我们需要导入tensorflow模块。
然后定义添加神经层的函数def add_layer(),它有四个参数：输入值、输入的大小、输出的大小和激励函数，我们设定默认的激励函数是None。

    def add_layer(inputs, in_size, out_size, activation_function=None):    

接下来，我们开始定义weights和biases。
因为在生成初始参数时，随机变量(normal distribution)会比全部为0要好很多，所以我们这里的weights为一个in_size行, out_size列的随机变量矩阵。
    
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))

在机器学习中，biases的推荐值不为0，所以我们这里是在0向量的基础上又加了0.1。

    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)

下面，我们定义Wx_plus_b, 即神经网络未激活的值。其中，tf.matmul()是矩阵的乘法。

    Wx_plus_b = tf.matmul(inputs, Weights) + biases

当activation_function——激励函数为None时，输出就是当前的预测值——Wx_plus_b，不为None时，就把Wx_plus_b传到activation_function()函数中得到输出。

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)

最后，返回输出，添加一个神经层的函数——def add_layer()就定义好了。

    return outputs
