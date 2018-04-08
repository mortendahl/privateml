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

def generate_dataset(size, num_features):
    assert size % 2 == 0

    np.random.seed(42)

    # X0 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], size//2)
    # X1 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], size//2)
    # X = np.vstack((X0, X1)).astype(np.float32)

    X = np.random.random_sample((size, num_features))

    Y0 = np.zeros(size//2).reshape(-1, 1)
    Y1 = np.ones (size//2).reshape(-1, 1)
    Y = np.vstack((Y0, Y1)).astype(np.float32)

    # shuffle
    perm = np.random.permutation(len(X))
    X = X[perm]
    Y = Y[perm]

    return X, Y

##############################
#       Public training      #
##############################

def public_training(X, Y, batch_size=100, learning_rate=0.01, epochs=10):

    data_size, num_features = X.shape
    assert data_size % batch_size == 0

    num_batches = data_size // batch_size

    batches_X = [ X[batch_size*batch_index : batch_size*(batch_index+1)] for batch_index in range(num_batches) ]
    batches_Y = [ Y[batch_size*batch_index : batch_size*(batch_index+1)] for batch_index in range(num_batches) ]

    W = np.zeros((num_features, 1))
    B = np.zeros((1, 1))

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

##############################
#      Public prediction     #
##############################

def setup_public_prediction(W, B, shape_X):

    def public_prediction(X):
        X = X.reshape(shape_X)
        return np_sigmoid(np.dot(X, W) + B)

    return public_prediction

##############################
#     Private prediction     #
##############################

# def prepare_input(inputs):
#     for input, tensor:
#         dict(
#             (input_xi, Xi) for input_xi, Xi in zip(input_x, decompose(encode(X)))
#         )

# def decode_output(output):
#   return decode(recombine(output))

def setup_private_prediction(W, B, shape_X):

    input_x, x = define_input(shape_X, name='x')

    w = define_variable(W, name='w')
    b = define_variable(B, name='b')

    # w = cache(mask(w))
    # b = cache(b)
    y = sigmoid(add(dot(x, w), b))

    def private_prediction(X):
        X = X.reshape(shape_X)

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
                tf.global_variables_initializer(),
                options=run_options,
                run_metadata=run_metadata
            )
            write_metadata('init')

            sess.run(
                cache_updators,
                options=run_options,
                run_metadata=run_metadata
            )
            write_metadata('populate')

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

                y_pred_private = decode(recombine(y_pred))

            writer.close()

            return y_pred_private

    return private_prediction

##############################
#          Running           #
##############################

X, Y = generate_dataset(10000, 100)
W, B = public_training(X, Y)

# predict on batch size of 100
private_input = X[:100]

public_prediction  = setup_public_prediction(W, B, shape_X=private_input.shape)
private_prediction = setup_private_prediction(W, B, shape_X=private_input.shape)

y_pred_public  = public_prediction(private_input)
y_pred_private = private_prediction(private_input)

print y_pred_public[:10], len(y_pred_public)
print y_pred_private[:10], len(y_pred_private)

assert (abs(y_pred_private - y_pred_public) < 0.01).all(), y_pred_private
