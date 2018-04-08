import tensorflow as tf

SESSION_CONFIG = tf.ConfigProto(
    log_device_placement=True,
    allow_soft_placement=False
)

JOB_NAME = 'spdz'

SERVER_0 = '/job:{}/replica:0/task:0/cpu:0'.format(JOB_NAME)
SERVER_1 = '/job:{}/replica:0/task:1/cpu:0'.format(JOB_NAME)
CRYPTO_PRODUCER = '/job:{}/replica:0/task:2/cpu:0'.format(JOB_NAME)
INPUT_PROVIDER  = '/job:{}/replica:0/task:3/cpu:0'.format(JOB_NAME)
OUTPUT_RECEIVER = '/job:{}/replica:0/task:4/cpu:0'.format(JOB_NAME)

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

TENSORBOARD_DIR = '/tmp/tensorboard'

session = lambda: tf.Session(MASTER, config=SESSION_CONFIG)