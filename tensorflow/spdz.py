
# TODOs:
# - recombine without blowing up numbers (should fit in 64bit word)
# - fixedpoint encoding
# - truncation operation (using CRT mod operation)
# - dot product
# - gradient computation
# - SDG loop
# - compare performance if native type is float64 instead of int64
# - performance on GPU
# - recombine without blowing up numbers

from datetime import datetime
from functools import reduce
from math import log

import numpy as np
import tensorflow as tf

log2 = lambda x: log(x)/log(2)
prod = lambda xs: reduce(lambda x,y: x*y, xs)


# Idea is to simulate five different players on different devices. 
# Hopefully Tensorflow can take care of (optimising?) networking like this.

SERVER_0 = '/device:CPU:0'
SERVER_1 = '/device:CPU:1'
CRYPTO_PRODUCER = '/device:CPU:2'
INPUT_PROVIDER  = '/device:CPU:3'
OUTPUT_RECEIVER = '/device:CPU:4'

config = tf.ConfigProto(
    log_device_placement=True,
    device_count={"CPU": 5},
    inter_op_parallelism_threads=1,
    intra_op_parallelism_threads=1
)

DOT_OP = tf.matmul

# NATIVE_TYPE = tf.float64 # TODO didn't work out of the box


#
# 64 bit CRT
# - 5 components for modulus ~120 bits for encoding 16.32
#

# NATIVE_TYPE = tf.int64
# m = [89702869, 78489023, 69973811, 70736797, 79637461]
# M = 2775323292128270996149412858586749843569 # == prod(m)
# assert log2(M) >= 120
# for mi in m: assert 2*log2(mi) + log2(1024) < 63
# lambdas = [
#     875825745388370376486957843033325692983, 
#     2472444909335399274510299476273538963924, 
#     394981838173825602426191615931186905822, 
#     2769522470404025199908727961712750149119, 
#     1813194913083192535116061678809447818860
# ]

#
# 32 bit CRT
# - 10 components for modulus ~100 bits for encoding 10.22
# - we need this to do dot product as int32 is the only supported type for that
#

NATIVE_TYPE = tf.int32
m = [1201, 1433, 1217, 1237, 1321, 1103, 1129, 1367, 1093, 1039]
M = 6616464272061971915798970247351 # == prod(m)
assert log2(M) >= 100
for mi in m: assert 2*log2(mi) + log2(1024) < 31
lambdas = [
    1008170659273389559193348505633, 
    678730110253391396805616626909, 
    3876367317978788805229799331439, 
    1733010852181147049893990590252, 
    2834912019672275627813941831946, 
    5920625781074493455025914446179, 
    4594604064921688203708053741296, 
    4709451160728821268524065874669,
    4618812662015813880836792588041, 
    3107636732210050331963327700392
]


def decompose(x):
    return tuple( x % mi for mi in m )

def recombine(x):
    return sum( xi * li for xi, li in zip(x, lambdas) ) % M

# *** NOTE ***
# keeping mod operations in-lined here for simplicity;
# we should do them lazily

def crt_add(x, y):
    with tf.name_scope("crt_add"):
        return [ (xi + yi) % mi for xi, yi, mi in zip(x, y, m) ]

def crt_sub(x, y):
    with tf.name_scope("crt_sub"):
        return [ (xi - yi) % mi for xi, yi, mi in zip(x, y, m) ]

def crt_mul(x, y):
    with tf.name_scope("crt_mul"):
        return [ (xi * yi) % mi for xi, yi, mi in zip(x, y, m) ]

def crt_dot(x, y):
    with tf.name_scope("crt_dot"):
        return [ DOT_OP(xi, yi) % mi for xi, yi, mi in zip(x, y, m) ]

class PrivateVariable:
    
    def __init__(self, share0, share1):
        self.share0 = share0
        self.share1 = share1
        
    def __add__(x, y):
        return add(x, y)
    
    def __sub__(x, y):
        return sub(x, y)
    
    def __mul__(x, y):
        return mul(x, y)

def sample(shape):
    with tf.name_scope("sample"):
        return [ tf.random_uniform(shape, maxval=mi, dtype=NATIVE_TYPE) for mi in m ]

def share(secret):
    with tf.name_scope("share"):
        shape = secret[0].shape
        share0 = sample(shape)
        share1 = crt_sub(secret, share0)
        return share0, share1

def reconstruct(share0, share1):
    with tf.name_scope("reconstruct"):
        return crt_add(share0, share1)

def add(x, y):
    assert isinstance(x, PrivateVariable)
    assert isinstance(y, PrivateVariable)
    
    x0, x1 = x.share0, x.share1
    y0, y1 = y.share0, y.share1
    
    with tf.name_scope("add"):
    
        with tf.device(SERVER_0):
            z0 = crt_add(x0, y0)

        with tf.device(SERVER_1):
            z1 = crt_add(x1, y1)

    return PrivateVariable(z0, z1)

def sub(x, y):
    assert isinstance(x, PrivateVariable)
    assert isinstance(y, PrivateVariable)
    
    x0, x1 = x.share0, x.share1
    y0, y1 = y.share0, y.share1
    
    with tf.name_scope("sub"):
    
        with tf.device(SERVER_0):
            z0 = crt_sub(x0, y0)

        with tf.device(SERVER_1):
            z1 = crt_sub(x1, y1)

    return PrivateVariable(z0, z1)

def mul(x, y):
    assert isinstance(x, PrivateVariable)
    assert isinstance(y, PrivateVariable)
    
    x0, x1 = x.share0, x.share1
    y0, y1 = y.share0, y.share1

    with tf.name_scope("mul"):
    
        with tf.device(CRYPTO_PRODUCER):
            a = sample((10,10))
            b = sample((10,10))
            ab = crt_mul(a, b)

            a0, a1 = share(a)
            b0, b1 = share(b)
            ab0, ab1 = share(ab)

        with tf.device(SERVER_0):
            alpha0 = crt_sub(x0, a0)
            beta0  = crt_sub(y0, b0)

        with tf.device(SERVER_1):
            alpha1 = crt_sub(x1, a1)
            beta1  = crt_sub(y1, b1)

        # exchange of alpha's and beta's

        with tf.device(SERVER_0):
            alpha = reconstruct(alpha0, alpha1)
            beta = reconstruct(beta0, beta1)
            z0 = crt_add(ab0,
                 crt_add(crt_mul(a0, beta),
                 crt_add(crt_mul(alpha, b0),
                         crt_mul(alpha, beta))))

        with tf.device(SERVER_1):
            alpha = reconstruct(alpha0, alpha1)
            beta = reconstruct(beta0, beta1)
            z1 = crt_add(ab1,
                 crt_add(crt_mul(a1, beta),
                         crt_mul(alpha, b1)))
        
    return PrivateVariable(z0, z1)

def dot(x, y):
    assert isinstance(x, PrivateVariable)
    assert isinstance(y, PrivateVariable)
    
    x0, x1 = x.share0, x.share1
    y0, y1 = y.share0, y.share1

    with tf.name_scope("dot"):
    
        with tf.device(CRYPTO_PRODUCER):
            a = sample((10,10))
            b = sample((10,10))
            ab = crt_dot(a, b)

            a0, a1 = share(a)
            b0, b1 = share(b)
            ab0, ab1 = share(ab)

        with tf.device(SERVER_0):
            alpha0 = crt_sub(x0, a0)
            beta0  = crt_sub(y0, b0)

        with tf.device(SERVER_1):
            alpha1 = crt_sub(x1, a1)
            beta1  = crt_sub(y1, b1)

        # exchange of alpha's and beta's

        with tf.device(SERVER_0):
            alpha = reconstruct(alpha0, alpha1)
            beta = reconstruct(beta0, beta1)
            z0 = crt_add(ab0,
                 crt_add(crt_dot(a0, beta),
                 crt_add(crt_dot(alpha, b0),
                         crt_dot(alpha, beta))))

        with tf.device(SERVER_1):
            alpha = reconstruct(alpha0, alpha1)
            beta = reconstruct(beta0, beta1)
            z1 = crt_add(ab1,
                 crt_add(crt_dot(a1, beta),
                         crt_dot(alpha, b1)))
        
    return PrivateVariable(z0, z1)

def define_input(shape):
    
    with tf.name_scope("input"):
        
        with tf.device(INPUT_PROVIDER):
            input_x = [ tf.placeholder(NATIVE_TYPE, shape=shape) for _ in range(len(m)) ]
            x = share(input_x)
        
    return input_x, PrivateVariable(*x)

def define_variable(shape):
    
    with tf.device(SERVER_0):
        x0 = [ tf.Variable(tf.ones(shape=shape, dtype=tf.int64)) for _ in range(len(m)) ]

    with tf.device(SERVER_1):
        x1 = [ tf.Variable(tf.ones(shape=shape, dtype=tf.int64)) for _ in range(len(m)) ]
        
    return PrivateVariable(x0, x1)

def reveal(x):
    assert isinstance(x, PrivateVariable)
    
    x0, x1 = x.share0, x.share1

    with tf.name_scope("reveal"):
    
        with tf.device(OUTPUT_RECEIVER):
            y = reconstruct(x0, x1)
    
    return y

#
# Simple addition
#

# Inputs
input_x, x = define_input((10,10))
input_w, w = define_input((10,10))
input_b, b = define_input((10,10))

# Computation
z = reveal(dot(x, w) + b)

# Actual inputs
X = np.array([2**80 for _ in range(10*10)]).reshape((10,10)) % M
W = np.array([    2 for _ in range(10*10)]).reshape((10,10)) % M
B = np.array([    0 for _ in range(10*10)]).reshape((10,10)) % M


# Decomposed values outside Tensorflow
inputs = dict(
    [ (xi, Xi) for xi, Xi in zip(input_x, decompose(X)) ] +
    [ (wi, Wi) for wi, Wi in zip(input_w, decompose(W)) ] +
    [ (bi, Bi) for bi, Bi in zip(input_b, decompose(B)) ]
)

# Run computation using Tensorflow
with tf.Session(config=config) as sess:

    writer = tf.summary.FileWriter("/tmp/tensorflow", sess.graph)
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    sess.run(tf.global_variables_initializer())
    
    res = sess.run(
        z, 
        inputs,
        options=run_options,
        run_metadata=run_metadata
    )
    
    writer.add_run_metadata(run_metadata, '')
    writer.close()

# Recombine result outside Tensorflow
Z = recombine(res)


assert (Z == (np.dot(X, W) + B)).all(), Z