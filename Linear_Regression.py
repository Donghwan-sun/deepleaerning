import tensorflow as tf

'''
    hypothesis
    H(x) = W * x + b 
'''
x = [1, 2, 3]
y = [1, 2, 3]
W = tf.Variable(tf.random.normal([1]), name="weight")
b = tf.Variable(tf.random.normal([1]), name="bias ")

