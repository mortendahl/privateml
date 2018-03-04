#!/usr/bin/env python2

from datetime import datetime

from config import MASTER, SESSION_CONFIG, TENSORBOARD_DIR
from tensorspdz import *

# Inputs
input_x, x = define_input((100,100))
input_w, w = define_input((100,100))
input_b, b = define_input((100,100))

# Computation
y = dot(x, w) + b
z = reveal(y)

# Actual inputs
X = np.random.randn(100,100)
W = np.random.randn(100,100)
B = np.random.randn(100,100)

# Decomposed values outside Tensorflow
inputs = dict(
    [ (xi, Xi) for xi, Xi in zip(input_x, decompose(encode(X))) ] +
    [ (wi, Wi) for wi, Wi in zip(input_w, decompose(encode(W))) ] +
    [ (bi, Bi) for bi, Bi in zip(input_b, decompose(encode(B))) ]
)

# Run computation using Tensorflow
with tf.Session(MASTER, config=SESSION_CONFIG) as sess:

    writer = tf.summary.FileWriter(TENSORBOARD_DIR, sess.graph)
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    sess.run(tf.global_variables_initializer())
    
    start = datetime.now()
    for i in range(10):
        res = sess.run(
            z,
            inputs,
            options=run_options,
            run_metadata=run_metadata
        )
        writer.add_run_metadata(run_metadata, 'run-{}'.format(i))
    end = datetime.now()

    print(end - start)

    writer.close()

# Recover result outside Tensorflow
Z = decode(recombine(res))

expected = np.dot(X, W) + B
actual   = Z
diff = expected - actual
assert (abs(diff) < 1e-3).all(), diff