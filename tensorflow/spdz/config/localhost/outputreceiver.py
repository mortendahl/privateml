#!/usr/bin/env python2

import tensorflow as tf
from config import CLUSTER, JOB_NAME

server = tf.train.Server(CLUSTER, job_name=JOB_NAME, task_index=4)
server.start()
server.join()