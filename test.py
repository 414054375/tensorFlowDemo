import tensorflow as tf
import numpy
rng = numpy.random

X = tf.constant([1, 2, 3, 4], dtype="float32")
Y = tf.constant([2, 3, 4, 5], dtype="float32")
train_x = numpy.asarray([1, 2, 3, 4])
train_y = numpy.asarray([2, 3, 4, 5])
x_input = tf.placeholder("float")
y_input = tf.placeholder("float")


w = tf.Variable(2.0, name="weight")
b = tf.Variable(2.0, name="bias")
w_input = tf.Variable(2.0, name="weight_in")
b_input = tf.Variable(2.0, name="bias_in")

pred = tf.add(tf.multiply(X, w), b)
pred_input = tf.add(tf.multiply(x_input, w_input), b_input)
cost = tf.reduce_sum(tf.pow(pred-Y, 2) / 4)
cost_input = tf.reduce_sum(tf.pow(pred_input-Y, 2) / 4)
optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
optimizer_input = tf.train.GradientDescentOptimizer(0.01).minimize(cost_input)
init = tf.global_variables_initializer()
with tf.Session() as sess:

    sess.run(init)
    for epoch in range(200):
        print ("w: %f  b: %f" % (sess.run(w), sess.run(b)))
        sess.run(optimizer)
        print ("After run optimizer:", "w: %f  b: %f" % (sess.run(w), sess.run(b)))
    '''
    print ("input w: %f  b: %f" % (sess.run(w_input), sess.run(b_input)))
    sess.run(optimizer_input, feed_dict={x_input:train_x, y_input:train_y})
    print ("After run optimizer:", " input w: %f  b: %f" % (sess.run(w), sess.run(b)))
    '''


