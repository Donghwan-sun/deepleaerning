import tensorflow as tf
import numpy as np


'''
    H(x) = W * x + b 
'''
x = [1, 2, 3]
y = [1, 2, 3]
W = tf.Variable(tf.random.normal([1]), name="weight")
b = tf.Variable(tf.random.normal([1]), name="bias ")

#hypothesis = x * W +b
hypothesis = x * W + b

#cost 코드 구현
cost = tf.reduce_mean(tf.square(hypothesis-y))

#모두의 딥러닝 tf2.0 linear_regression 구현코드
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

tf.model = tf.keras.Sequential()
# units == output shape, input_dim == input shape
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=1))

sgd = tf.keras.optimizers.SGD(lr=0.1)  # SGD == standard gradient descendent, lr == learning rate
tf.model.compile(loss='mse', optimizer=sgd)  # mse == mean_squared_error, 1/m * sig (y'-y)^2

# prints summary of the model to the terminal
tf.model.summary()

# fit() executes training
tf.model.fit(x_train, y_train, epochs=200)

# predict() returns predicted value
y_predict = tf.model.predict(np.array([5, 4]))
print(y_predict)