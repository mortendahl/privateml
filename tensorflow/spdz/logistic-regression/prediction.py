#!/usr/bin/env python2

# hack to import tensorspdz from parent directory
# - https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))


from datetime import datetime

from config import session, TENSORBOARD_DIR
from tensorspdz import *

from tensorflow.python.client import timeline


np_sigmoid = lambda x: 1 / (1 + np.exp(-x))

##############################
#   Generate training data   #
##############################

np.random.seed(42)

data_size = 10000
assert data_size % 2 == 0

X0 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], data_size//2)
X1 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], data_size//2)
X = np.vstack((X0, X1)).astype(np.float32)

Y0 = np.zeros(data_size//2).reshape(-1, 1)
Y1 = np.ones (data_size//2).reshape(-1, 1)
Y = np.vstack((Y0, Y1)).astype(np.float32)

# shuffle
perm = np.random.permutation(len(X))
X = X[perm]
Y = Y[perm]


##############################
#       Public training      #
##############################

def public_training():

    batch_size = 100
    assert data_size % batch_size == 0
    num_batches = data_size // batch_size

    batches_X = [ X[batch_size*batch_index : batch_size*(batch_index+1)] for batch_index in range(num_batches) ]
    batches_Y = [ Y[batch_size*batch_index : batch_size*(batch_index+1)] for batch_index in range(num_batches) ]

    shape_x = (batch_size,2)
    shape_y = (batch_size,1)
    shape_w = (2,1)
    shape_b = (1,1)

    learning_rate = 0.01
    epochs = 10

    W = np.zeros(shape_w)
    B = np.zeros(shape_b)

    for _ in range(epochs):

        for X_batch, Y_batch in zip(batches_X, batches_Y):

            # forward
            Y_pred = np_sigmoid(np.dot(X_batch, W) + B)

            # backward
            error = Y_pred - Y_batch
            d_W = np.dot(X_batch.transpose(), error) * 1./batch_size
            d_B = np.sum(error, axis=0) * 1./batch_size
            W = W - d_W * learning_rate
            B = B - d_B * learning_rate

    preds = np.round(np_sigmoid(np.dot(X, W) + B))
    accuracy = (preds == Y).sum().astype(float) / len(preds)
    print 'accuracy:', accuracy

    return W, B

W, B = public_training()


##############################
#      Public prediction     #
##############################

def setup_public_prediction(W, B):

    def public_prediction(X):
        X = X.reshape(-1, 2)
        return np_sigmoid(np.dot(X, W) + B)

    return public_prediction

public_prediction = setup_public_prediction(W, B)

print public_prediction(X[:10])


##############################
#     Private prediction     #
##############################

def setup_private_prediction(W, B, shape_X=(1,2)):

    input_x, x = define_input(shape_X, name='x')

    init_w, w = define_variable(W, name='w')
    init_b, b = define_variable(B, name='b')

    y = sigmoid(add(dot(x, w), b))

    def private_prediction(X):
        X = X.reshape(-1, 2)

        with session() as sess:

            writer = tf.summary.FileWriter(TENSORBOARD_DIR, sess.graph)
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            def write_metadata(tag):
                writer.add_run_metadata(run_metadata, tag)
                chrome_trace = timeline.Timeline(run_metadata.step_stats).generate_chrome_trace_format()
                with open('{}/{}.ctr'.format(TENSORBOARD_DIR, tag), 'w') as f:
                    f.write(chrome_trace)

            sess.run(
                [init_w, init_b],
                options=run_options,
                run_metadata=run_metadata
            )
            write_metadata('setup')

            for i in range(10):

                y_pred = sess.run(
                    reveal(y),
                    feed_dict=dict(
                        (input_xi, Xi) for input_xi, Xi in zip(input_x, decompose(encode(X)))
                    ),
                    options=run_options,
                    run_metadata=run_metadata
                )
                write_metadata('predict-{}'.format(i))
                
            writer.close()

            return decode(recombine(y_pred))

    return private_prediction

private_prediction = setup_private_prediction(W, B, shape_X=(10,2))

print private_prediction(X[:10])
