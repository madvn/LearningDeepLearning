##########################################################################
# Pretty much same as tutorial on TF webpage
# Created Aug 29, 2017
# Madhavun Candadai
##########################################################################
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

### Computational graph definition starts here ###

# placeholder for data (things that are not learnt)
x = tf.placeholder(tf.float32, [None, 784])

# Variables for things are being learnt and need to be inited
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# prediction and actual output
yhat = (tf.matmul(x,W)+b) # requires to be softmaxed to be called prediction though 
y = tf.placeholder(tf.float32, [None,10])

# Computing loss and setting up learning
#cross_entropy = tf.reduce_mean(-tf.reduce_mean(y * tf.log(yhat), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

# proportion of correct classification (note reusing yhat and the placeholder y but with test data)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,1),tf.argmax(yhat,1)),tf.float32))

### Computational graph definition ends here ###

### Now on to executing the computational graph ###

# default TF session
sess = tf.InteractiveSession()

# initing weights and bias
tf.global_variables_initializer().run()

# read data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# train
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x:batch_xs,y:batch_ys})

# See performance
print "Accuracy = ", sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
