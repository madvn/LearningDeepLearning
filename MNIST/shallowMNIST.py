import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


x = tf.placeholder(tf.float32, [None, 784])

w1 = tf.Variable(tf.zeros([784,6000]))
b1 = tf.Variable(tf.zeros([1,6000]))
y1 = tf.sigmoid(tf.matmul(x,w1)+b1)

'''
w2 = tf.Variable(tf.zeros([784,784]))
b2 = tf.Variable(tf.zeros([1,784]))
y2 = tf.sigmoid(tf.matmul(y1,w2)+b2)

w3 = tf.Variable(tf.zeros([784,784]))
b3 = tf.Variable(tf.zeros([1,784]))
y3 = tf.sigmoid(tf.matmul(y2,w3)+b3)
'''

w4a = tf.Variable(tf.zeros([6000,10]))
w4b = tf.Variable(tf.zeros([784,10]))
b4 = tf.Variable(tf.zeros([1,10]))
yhat = (tf.matmul(y1,w4a)+tf.matmul(x,w4b)+b4)

y = tf.placeholder(tf.float32,[None, 10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yhat, labels=y))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(yhat,1),tf.argmax(y,1)), tf.float32))



mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(10000):
        xd, yd = mnist.train.next_batch(100)
        if i%1000 == 0:
            print 'Training Perf @', i ,' = ', accuracy.eval({x:xd,y:yd})
        sess.run(train_step, feed_dict={x:xd,y:yd})

    print 'Accuracy = ', accuracy.eval({x:mnist.test.images,y:mnist.test.labels})
