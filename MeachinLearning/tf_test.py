import tensorflow as tf
import numpy as np

#creat data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

Weights = tf.Variable(tf.random_uniform([1], -1.0))
bias = tf.Variable(tf.zeros([1]))

y = Weights * x_data + bias

loss = tf.reduce_mean(tf.square(y-y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)

train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(bias))
sess.close()

##sess
matrix = tf.constant([[3,3]])
matrix2 = tf.constant([[2],[2]])
product = tf.matmul(matrix, matrix2)
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)

##Variables
state = tf.Variable(0,name = 'counter')
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)
init = tf.initialize_all_variables() #must have if define variables

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

print(state.name)

##Add layer
import numpy as np
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs
x_data = np.linspace(-1,1,300)[:, np.newaxis]
noise = np.random.normal(0, 0.005, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
l1 = add_layer(xs,1,10,activation_function = tf.nn.relu)
prediction = add_layer(l1,10,1, activation_function = None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
    if i % 50:
        print(sess.run(loss, feed_dict = {xs: x_data, ys: y_data}))