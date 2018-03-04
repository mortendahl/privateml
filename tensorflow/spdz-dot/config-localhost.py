import tensorflow as tf

SESSION_CONFIG = tf.ConfigProto(
    log_device_placement=True
)

JOB_NAME = 'spdz'

CRYPTO_PRODUCER = '/job:{}/task:2'.format(JOB_NAME)
INPUT_PROVIDER  = '/job:{}/task:3'.format(JOB_NAME)
SERVER_0 = '/job:{}/task:0'.format(JOB_NAME)
SERVER_1 = '/job:{}/task:1'.format(JOB_NAME)
OUTPUT_RECEIVER = '/job:{}/task:4'.format(JOB_NAME)

HOSTS = [
    'localhost:4440',
    'localhost:4441',
    'localhost:4442',
    'localhost:4443',
    'localhost:4444'
]

MASTER = 'grpc://{}'.format(HOSTS[0])

CLUSTER = tf.train.ClusterSpec({
    JOB_NAME: HOSTS
})

TENSORBOARD_DIR = '/tmp/tensorflow'