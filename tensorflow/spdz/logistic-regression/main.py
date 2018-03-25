#!/usr/bin/env python2

# hack to import tensorspdz from parent directory
# - https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder
# import sys, os
# sys.path.insert(1, os.path.join(sys.path[0], '..'))


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

# augment x with all-one column
B = np.ones(data_size).reshape(-1, 1)
X = np.hstack((X, B))

Y0 = np.zeros(data_size//2).reshape(-1, 1)
Y1 = np.ones (data_size//2).reshape(-1, 1)
Y = np.vstack((Y0, Y1)).astype(np.float32)

# shuffle
perm = np.random.permutation(len(X))
X = X[perm]
Y = Y[perm]

batch_size = 100
assert data_size % batch_size == 0
num_batches = data_size // batch_size

batches_x = [ X[batch_size*batch_index : batch_size*(batch_index+1)] for batch_index in range(num_batches) ]
batches_y = [ Y[batch_size*batch_index : batch_size*(batch_index+1)] for batch_index in range(num_batches) ]

shape_x = (batch_size,3)
shape_y = (batch_size,1)
shape_w = (3,1)

##############################
#          training          #
##############################

learning_rate = 0.01
epochs = 10

def accuracy(w):
    preds = np.round(np_sigmoid(np.dot(X, w)))
    return (preds == Y).sum().astype(float) / len(preds)

##############################
#       Public training      #
##############################

def public_training():

    w = np.zeros(shape_w)

    start = datetime.now()

    for _ in range(epochs):
        for x, y in zip(batches_x, batches_y):
            # forward
            y_pred = np_sigmoid(np.dot(x, w))
            # backward
            error = y_pred - y
            gradients = np.dot(x.transpose(), error) * 1./batch_size
            w = w - gradients * learning_rate

    end = datetime.now()
    print end-start

    print w, accuracy(w)

    return w

##############################
#      Private training      #
##############################

def define_batch_buffer(shape, capacity):

    # each server holds three values: xi, ai, and alpha
    server_packed_shape = (3, len(m),) + tuple(shape)

    # the crypto producer holds just one value: a
    cryptoprovider_packed_shape = (len(m),) + tuple(shape)

    with tf.device(SERVER_0):
        queue_0 = tf.FIFOQueue(
            capacity=capacity,
            dtypes=[INT_TYPE],
            shapes=[server_packed_shape]
        )

    with tf.device(SERVER_1):
        queue_1 = tf.FIFOQueue(
            capacity=capacity,
            dtypes=[INT_TYPE],
            shapes=[server_packed_shape]
        )

    with tf.device(CRYPTO_PRODUCER):
        queue_cp = tf.FIFOQueue(
            capacity=capacity,
            dtypes=[INT_TYPE],
            shapes=[cryptoprovider_packed_shape]
        )

    return (queue_0, queue_1, queue_cp)

def distribute_batch(shape, buffer):

    queue_0, queue_1, queue_cp = buffer

    def pack_server(tensors):
        with tf.name_scope('pack'):
            return tf.stack([ tf.stack(tensor, axis=0) for tensor in tensors ], axis=0)

    def pack_cryptoproducer(tensor):
        with tf.name_scope('pack'):
            return tf.stack(tensor, axis=0)

    with tf.name_scope('enqueue'):

        input_x = [ tf.placeholder(INT_TYPE, shape=shape) for _ in m ]

        # share x
        x0, x1 = share(input_x)
        
        # precompute mask
        a = sample(shape)
        a0, a1 = share(a)
        alpha = crt_sub(input_x, a)

        with tf.device(SERVER_0):
            enqueue_0 = queue_0.enqueue(pack_server([x0, a0, alpha]))

        with tf.device(SERVER_1):
            enqueue_1 = queue_1.enqueue(pack_server([x1, a1, alpha]))

        with tf.device(CRYPTO_PRODUCER):
            enqueue_cp = queue_cp.enqueue(pack_cryptoproducer(a))

        populate_op = [enqueue_0, enqueue_1, enqueue_cp]

    return input_x, populate_op

def load_batch(shape, buffer):

    shape = tuple(shape)
    queue_0, queue_1, queue_cp = buffer

    def unpack_server(tensors):
        with tf.name_scope('unpack'):
            return [
                [
                    tf.reshape(subtensor, shape)
                    for subtensor in tf.split(tf.reshape(tensor, (len(m),) + shape), len(m))
                ]
                for tensor in tf.split(tensors, 3)
            ]

    def unpack_cryptoproducer(tensor):
        with tf.name_scope('unpack'):
            return [
                tf.reshape(subtensor, shape)
                for subtensor in tf.split(tensor, len(m))
            ]

    with tf.name_scope('dequeue'):

        with tf.device(SERVER_0):
            packed_0 = queue_0.dequeue()
            reenqueue_0 = queue_0.enqueue(packed_0)
            x0, a0, alpha_on_0 = unpack_server(packed_0)

        with tf.device(SERVER_1):
            packed_1 = queue_1.dequeue()
            reenqueue_1 = queue_1.enqueue(packed_1)
            x1, a1, alpha_on_1 = unpack_server(packed_1)

        with tf.device(CRYPTO_PRODUCER):
            packed_cp = queue_cp.dequeue()
            reenqueue_cp = queue_cp.enqueue(packed_cp)
            a = unpack_cryptoproducer(packed_cp)

    x = PrivateTensor(x0, x1)
    
    cache_key = ('mask', x)
    cached_results[cache_key] = (a, a0, a1, alpha_on_0, alpha_on_1)

    return x, [reenqueue_0, reenqueue_1, reenqueue_cp]

def training_loop(buffers, shapes, iterations, initial_weights, training_step):

    buffer_x, buffer_y = buffers
    shape_x, shape_y = shapes

    initial_w0, initial_w1 = share(initial_weights)

    # TODO re-enqueue using `re_x` and `re_y`
    def loop_op(w0, w1):
        w = PrivateTensor(w0, w1)
        x, re_x = load_batch(shape_x, buffer_x)
        y, re_y = load_batch(shape_y, buffer_y)
        new_w = training_step(w, x, y)
        return new_w.share0, new_w.share1

    _, final_w0, final_w1 = tf.while_loop(
        cond=lambda i, w0, w1: tf.less(i, iterations),
        body=lambda i, w0, w1: (i+1,) + loop_op(w0, w1),
        loop_vars=(0, initial_w0, initial_w1),
        parallel_iterations=1
    )

    return final_w0, final_w1

buffer_x = define_batch_buffer(shape_x, num_batches)
buffer_y = define_batch_buffer(shape_y, num_batches)

input_x, distribute_x = distribute_batch(shape_x, buffer_x)
input_y, distribute_y = distribute_batch(shape_y, buffer_y)

def training_step(w, x, y):
    with tf.name_scope('forward'):
        y_pred = sigmoid(dot(x, w))
    with tf.name_scope('backward'):
        error = sub(y_pred, y)
        gradients = scale(dot(transpose(x), error), 1./batch_size)
        return sub(w, scale(gradients, learning_rate))

training = training_loop(
    buffers=(buffer_x, buffer_y),
    shapes=(shape_x, shape_y),
    iterations=num_batches,
    initial_weights=decompose(np.zeros(shape=shape_w)),
    training_step=training_step
)

##############################
#     Private prediction     #
##############################

def private_prediction(sess, w):

    INPUT_SIZE = 300

    input_x, x = define_input((INPUT_SIZE,3))
    
    y = reveal(sigmoid(dot(x, w)))

    writer = tf.summary.FileWriter(TENSORBOARD_DIR, sess.graph)
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    for i in range(10):

        Y = sess.run(
            y,
            feed_dict=dict(
                [ (xi, Xi) for xi, Xi in zip(input_x, decompose(encode(X[i:i+INPUT_SIZE].reshape(INPUT_SIZE,3)))) ]
            ),
            options=run_options,
            run_metadata=run_metadata
        )
        # print decode(recombine(Y))

        writer.add_run_metadata(run_metadata, 'prediction-{}'.format(i))

        chrome_trace = timeline.Timeline(run_metadata.step_stats).generate_chrome_trace_format()
        with open('{}/{}.ctr.json'.format(TENSORBOARD_DIR, 'prediction-{}'.format(i)), 'w') as f:
            f.write(chrome_trace)

    writer.close()

    return Y

##############################
#            Run             #
##############################

with session() as sess:

    writer = tf.summary.FileWriter(TENSORBOARD_DIR, sess.graph)
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    print 'Distributing...'
    for batch_x, batch_y in zip(batches_x, batches_y):

        sess.run(
            distribute_x,
            feed_dict=dict([
                (input_xi, Xi) for input_xi, Xi in zip(input_x, decompose(encode(batch_x)))
            ]),
            options=run_options,
            run_metadata=run_metadata
        )
        # writer.add_run_metadata(run_metadata, 'enqueue-x-{}'.format(i))

        sess.run(
            distribute_y,
            feed_dict=dict([
                (input_yi, Yi) for input_yi, Yi in zip(input_y, decompose(encode(batch_y)))
            ]),
            options=run_options,
            run_metadata=run_metadata
        )
        # writer.add_run_metadata(run_metadata, 'enqueue-y-{}'.format(i))

    print 'Training...'
    # start = datetime.now()
    w0, w1 = sess.run(
        training,
        options=run_options,
        run_metadata=run_metadata
    )
    # end = datetime.now()
    # print end-start
    # writer.add_run_metadata(run_metadata, 'train')

    # w = decode(recombine(reconstruct(w0, w1)))
    # print w, accuracy(w)

    # y = prediction(sess)
    
    writer.close()