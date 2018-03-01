import tensorflow as tf

from config import *

w = tf.constant(2)
b = tf.constant(2)

with tf.device(WORKER_0):
    x = tf.placeholder(tf.int32, shape=())
    y = x * w

with tf.device(WORKER_1):
    z = y + b

with tf.Session(MASTER) as sess:

    writer = tf.summary.FileWriter('/tmp/tensorflow', sess.graph)

    result = sess.run(z, {x: 2})
    print(result)

    writer.close()