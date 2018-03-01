#!/usr/bin/env python2

# TODOs:
# - truncation operation (using CRT mod operation)
# - recombine without blowing up numbers (should fit in 64bit word)
# - gradient computation + SGD
# - compare performance if native type is float64 instead of int64
# - performance on GPU

from datetime import datetime
from functools import reduce
from math import log

import numpy as np
import tensorflow as tf

log2 = lambda x: log(x)/log(2)
prod = lambda xs: reduce(lambda x,y: x*y, xs)


# Idea is to simulate five different players on different devices. 
# Hopefully Tensorflow can take care of (optimising?) networking like this.

#
# 64 bit CRT
# - 5 components for modulus ~120 bits (encoding 16.32)
#
# BITPRECISION_INTEGRAL   = 16
# BITPRECISION_FRACTIONAL = 32
# NATIVE_TYPE = tf.int64
# TRUNCATION_GAP = 20
# m = [89702869, 78489023, 69973811, 70736797, 79637461]
# M = 2775323292128270996149412858586749843569 # == prod(m)
# lambdas = [
#     875825745388370376486957843033325692983, 
#     2472444909335399274510299476273538963924, 
#     394981838173825602426191615931186905822, 
#     2769522470404025199908727961712750149119, 
#     1813194913083192535116061678809447818860
# ]

#
# 32 bit CRT
# - we need this to do dot product as int32 is the only supported type for that
# - tried tf.float64 but didn't work out of the box
# - 10 components for modulus ~100 bits
#
BITPRECISION_INTEGRAL   = 16
BITPRECISION_FRACTIONAL = 16
NATIVE_TYPE = tf.int32
TRUNCATION_GAP = 20
m = [1201, 1433, 1217, 1237, 1321, 1103, 1129, 1367, 1093, 1039]
M = 6616464272061971915798970247351 # == prod(m)
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

for mi in m: assert 2*log2(mi) + log2(1024) < log2(NATIVE_TYPE.max)
assert log2(M) >= 2 * (BITPRECISION_INTEGRAL + BITPRECISION_FRACTIONAL) + log2(1024) + TRUNCATION_GAP


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

TENSORBOARD_DIR = '/tmp/tensorflow'
DOT_OP = tf.matmul


def encode(rationals, precision=BITPRECISION_FRACTIONAL):
    return (rationals * (2**precision)).astype(int).astype(object) % M

def decode(elements, precision=BITPRECISION_FRACTIONAL):
    map_negative_range = np.vectorize(lambda element: element if element <= M/2 else element - M)
    return map_negative_range(elements).astype(float) / (2**precision)

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

# TODO: with truncation operation before adding b
# (and then we don't need double precision for B)

# Computation
y = dot(x, w) + b
z = reveal(y)

# Actual inputs
# X = np.array([2**80 for _ in range(10*10)]).reshape((10,10))
# W = np.array([    2 for _ in range(10*10)]).reshape((10,10))
# B = np.array([    0 for _ in range(10*10)]).reshape((10,10))
X = np.random.randn(10,10)
W = np.random.randn(10,10)
B = np.random.randn(10,10)

# Decomposed values outside Tensorflow
inputs = dict(
    [ (xi, Xi) for xi, Xi in zip(input_x, decompose(encode(X))) ] +
    [ (wi, Wi) for wi, Wi in zip(input_w, decompose(encode(W))) ] +
    [ (bi, Bi) for bi, Bi in zip(input_b, decompose(encode(B, precision=2*BITPRECISION_FRACTIONAL))) ]
)

# Run computation using Tensorflow
with tf.Session(config=config) as sess:

    writer = tf.summary.FileWriter(TENSORBOARD_DIR, sess.graph)
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

# Recover result outside Tensorflow
Z = decode(recombine(res), precision=2*BITPRECISION_FRACTIONAL)

assert (Z - (np.dot(X, W) + B) < 1e-3).all()