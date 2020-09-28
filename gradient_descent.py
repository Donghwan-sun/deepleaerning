import tensorflow as tf
import numpy

x = [1, 2, 3]
y = [1, 2, 3]
W = tf.Variable(tf.random.normal([1]), name="weight")
b = tf.Variable(tf.random.normal([1]), name="bias ")
learning_rate = 0.1
gradient = tf.reduce_mean((W * x - y) * x)
descent = W - learning_rate * gradient
update = W.assign(descent)
print(update)