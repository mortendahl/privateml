import tensorflow as tf

JOB_NAME = 'local'

CLUSTER = tf.train.ClusterSpec({
    JOB_NAME: ['localhost:4444', 'localhost:5555']
})

WORKER_0 = '/job:{}/task:0'.format(JOB_NAME)
WORKER_1 = '/job:{}/task:1'.format(JOB_NAME)

MASTER = 'grpc://localhost:4444'