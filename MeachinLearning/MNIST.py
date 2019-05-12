import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("Training data size:", mnist.train.num_examples)
print("Test data size:", mnist.test.num_examples)
x = tf.placeholder(tf.float32, [None, 784]) ##输入任意数量的MNIST图像，每一张图像784维度的向量，此处用2维度的浮点数张量来表示这些图，None 表示张量的第一个维度可以使任意长度。
y_ = tf.placeholder("float", [None, 10])##实际分类

##MLP: multilayer perceptron (多层感知器MLP,下面例子只用了1层)
##把每一个图形展开成一维向量输入网络中，忽略了图像的位置和结构信息。
# W = tf.Variable(tf.zeros([784, 10]))
# b = tf.Variable(tf.zeros([10]))
# y = tf.nn.softmax(tf.matmul(x,W) + b)
# cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# sess = tf.Session()
# #with tf.Session() as sess:
# sess.run(init)
# for i in range(1000):
#   batch = mnist.train.next_batch(100)
#   sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# print(sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))

##卷积神经网络（CNN）
##把图像以28*28的结构展开，更好地利用图像的位置和结构信息
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # 步长为1，strides = [batch, height, width, channle] 一般 batch 和 channel 都为1 （步长为1 不跳过 batch、channel）
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding='SAME')

##First layer
W_conv1 = weight_variable([5, 5, 1, 32]) # 5*5 with 1 input channel, 32 output channel
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1]) 

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

##Second layer
W_conv2 = weight_variable([5, 5, 32, 64]) ## 5*5 with 32 input channel, 64 output channel
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

##connect layer
##图片尺寸减小到7*7，加入一个有1024 个神经元的全连接层用于处理整个图片。把池化层输出的张量reshape 成一些向量，乘上权重加上偏置，然后对其ReLU.
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

##dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(session=sess, feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(session=sess, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(session=sess,feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

sess.close()