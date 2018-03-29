import tensorflow as tf
import json
import os

HOSTS_FILE = '/tmp/hosts.json'

SESSION_CONFIG = tf.ConfigProto(
    log_device_placement=True,
    allow_soft_placement=False
)

JOB_NAME = 'spdz'

# SERVER_0 = '/job:{}/task:0'.format(JOB_NAME)
# SERVER_1 = '/job:{}/task:1'.format(JOB_NAME)
# CRYPTO_PRODUCER = '/job:{}/task:2'.format(JOB_NAME)
# INPUT_PROVIDER  = '/job:{}/task:3'.format(JOB_NAME)
# OUTPUT_RECEIVER = '/job:{}/task:3'.format(JOB_NAME)

SERVER_0 = '/job:{}/replica:0/task:0/cpu:0'.format(JOB_NAME)
SERVER_1 = '/job:{}/replica:0/task:1/cpu:0'.format(JOB_NAME)
CRYPTO_PRODUCER = '/job:{}/replica:0/task:2/cpu:0'.format(JOB_NAME)
INPUT_PROVIDER  = '/job:{}/replica:0/task:3/cpu:0'.format(JOB_NAME)
OUTPUT_RECEIVER = '/job:{}/replica:0/task:3/cpu:0'.format(JOB_NAME)

with open(HOSTS_FILE, 'r') as f:
    HOSTS = json.load(f)

MASTER = 'grpc://{}'.format(HOSTS[0])

CLUSTER = tf.train.ClusterSpec({
    JOB_NAME: HOSTS
})

TENSORBOARD_DIR = '/tmp/tensorboard'

session = lambda: tf.Session(MASTER, config=SESSION_CONFIG)