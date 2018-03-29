#!/usr/bin/env python2

import sys
import tensorflow as tf
from config import CLUSTER, JOB_NAME

TASK = int(sys.argv[1])

server = tf.train.Server(CLUSTER, job_name=JOB_NAME, task_index=TASK)
server.start()
server.join()