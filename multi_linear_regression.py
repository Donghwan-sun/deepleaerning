import tensorflow as tf
import numpy as np
# tf 1.x 버전 사용시
x1 = [73, 93, 89, 96, 73]
x2 = [80, 88, 91, 98, 66]
x3 = [75, 93, 90, 100, 70]
y = [152, 185, 180, 196, 142]

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)

Y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random.normal[1], name='wegiht1')
w2 = tf.Variable(tf.random.normal[1], name='wegiht2')
w3 = tf.Variable(tf.random.normal[1], name='wegiht3')
b = tf.Variable(tf.random.normal[1], name='b')
# hypothesis

hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b
# cost

cost = tf.reduce_mean(tf.square(hypothesis - Y))
#minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)


# 모두의 딥러닝 tf2.0 버전

x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]

tf.model = tf.keras.Sequential()

tf.model.add(tf.keras.layers.Dense(units=1, input_dim=3))  # input_dim=3 gives multi-variable regression
tf.model.add(tf.keras.layers.Activation('linear'))  # this line can be omitted, as linear activation is default
# advanced reading https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6

tf.model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(lr=1e-5))
tf.model.summary()
history = tf.model.fit(x_data, y_data, epochs=100)

y_predict = tf.model.predict(np.array([[72., 93., 90.]]))
print(y_predict)
