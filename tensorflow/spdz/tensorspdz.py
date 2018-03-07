
# TODOs:
# - recombine without blowing up numbers (should fit in 64bit word)
# - gradient computation + SGD
# - compare performance if native type is float64 instead of int64
# - performance on GPU
# - better cache strategy?
# - does it make sense to cache additions, subtractions, etc as well?
# - make truncation optional; should work even with cached results

from functools import reduce
from math import log

import numpy as np
import tensorflow as tf

log2 = lambda x: log(x)/log(2)
prod = lambda xs: reduce(lambda x,y: x*y, xs)

from config import (
    SERVER_0, SERVER_1,
    CRYPTO_PRODUCER, INPUT_PROVIDER, OUTPUT_RECEIVER
)

# Idea is to simulate five different players on different devices. 
# Hopefully Tensorflow can take care of (optimising?) networking like this.

#
# 64 bit CRT
# - 5 components for modulus ~120 bits (encoding 16.32)
#
# BITPRECISION_INTEGRAL   = 16
# BITPRECISION_FRACTIONAL = 30
# INT_TYPE = tf.int64
# FLOAT_TYPE = tf.float64
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
INT_TYPE = tf.int32
FLOAT_TYPE = tf.float32
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

for mi in m: assert 2*log2(mi) + log2(1024) < log2(INT_TYPE.max)
assert log2(M) >= 2 * (BITPRECISION_INTEGRAL + BITPRECISION_FRACTIONAL) + log2(1024) + TRUNCATION_GAP

K = 2 ** BITPRECISION_FRACTIONAL

def egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)
    
def gcd(a, b):
    g, _, _ = egcd(a, b)
    return g

def inverse(a, m):
    _, b, _ = egcd(a, m)
    return b % m

def encode(rationals, precision=BITPRECISION_FRACTIONAL):
    """ [NumPy] Encode tensor of rational numbers into tensor of ring elements """
    return (rationals * (2**precision)).astype(int).astype(object) % M

def decode(elements, precision=BITPRECISION_FRACTIONAL):
    """ [NumPy] Decode tensor of ring elements into tensor of rational numbers """
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
        return [ tf.matmul(xi, yi) % mi for xi, yi, mi in zip(x, y, m) ]

def gen_crt_mod():

    # precomputation
    q = [ inverse(M // mi, mi) for mi in m ]
    B = M % K
    b = [ (M // mi) % K for mi in m ]

    def crt_mod(x):
        with tf.name_scope("crt_mod"):
            t = [ (xi * qi) % mi for xi, qi, mi in zip(x, q, m) ]
            alpha = tf.cast(
                tf.round(
                    tf.reduce_sum(
                        [ tf.cast(ti, FLOAT_TYPE) / mi for ti, mi in zip(t, m) ],
                        axis=0
                    )
                ),
                INT_TYPE
            )
            v = tf.reduce_sum(
                [ ti * bi for ti, bi in zip(t, b) ],
                axis=0
            ) - B * alpha
            return decompose(v % K)

    return crt_mod

crt_mod = gen_crt_mod()

def sample(shape):
    with tf.name_scope("sample"):
        return [ tf.random_uniform(shape, maxval=mi, dtype=INT_TYPE) for mi in m ]

def share(secret):
    with tf.name_scope("share"):
        shape = secret[0].shape
        share0 = sample(shape)
        share1 = crt_sub(secret, share0)
        return share0, share1

def reconstruct(share0, share1):
    with tf.name_scope("reconstruct"):
        return crt_add(share0, share1)

cached_results = dict()

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

    x_shape = x.shape
    y_shape = y.shape

    z = cached_results.get(('mul',x,y), None)

    if z is None:

        if x.masked is not None:
            a, a0, a1, alpha_on_0, alpha_on_1 = x.masked
        else:

            with tf.name_scope("masking"):

                with tf.device(CRYPTO_PRODUCER):
                    a = sample(x_shape)
                    a0, a1 = share(a)

                with tf.device(SERVER_0):
                    alpha0 = crt_sub(x0, a0)

                with tf.device(SERVER_1):
                    alpha1 = crt_sub(x1, a1)

                # exchange of alphas

                with tf.device(SERVER_0):
                    alpha = reconstruct(alpha0, alpha1)
                    alpha_on_0 = alpha # needed for saving in `masked`

                with tf.device(SERVER_1):
                    alpha = reconstruct(alpha0, alpha1)
                    alpha_on_1 = alpha # needed for saving in `masked`

            x.masked = (a, a0, a1, alpha_on_0, alpha_on_1)

        if y.masked is not None:
            b, b0, b1, beta_on_0, beta_on_1 = y.masked
        else:

            with tf.name_scope("masking"):

                with tf.device(CRYPTO_PRODUCER):
                    b = sample(y_shape)
                    b0, b1 = share(b)

                with tf.device(SERVER_0):
                    beta0 = crt_sub(y0, b0)

                with tf.device(SERVER_1):
                    beta1 = crt_sub(y1, b1)

                # exchange of betas

                with tf.device(SERVER_0):
                    beta = reconstruct(beta0, beta1)
                    beta_on_0 = beta # needed for saving in `masked`

                with tf.device(SERVER_1):
                    beta = reconstruct(beta0, beta1)
                    beta_on_1 = beta # needed for saving in `masked`

            y.masked = (b, b0, b1, beta_on_0, beta_on_1)

        with tf.name_scope("mul"):

            with tf.device(CRYPTO_PRODUCER):
                ab = crt_mul(a, b)
                ab0, ab1 = share(ab)

            with tf.device(SERVER_0):
                alpha = alpha_on_0
                beta = beta_on_0
                z0 = crt_add(ab0,
                    crt_add(crt_mul(a0, beta),
                    crt_add(crt_mul(alpha, b0),
                            crt_mul(alpha, beta))))

            with tf.device(SERVER_1):
                alpha = alpha_on_1
                beta = beta_on_1
                z1 = crt_add(ab1,
                    crt_add(crt_mul(a1, beta),
                            crt_mul(alpha, b1)))
        
        z = PrivateVariable(z0, z1)
        z = truncate(z)
        cached_results[('mul',x,y)] = z

    return z

def dot(x, y):
    assert isinstance(x, PrivateVariable)
    assert isinstance(y, PrivateVariable)

    x0, x1 = x.share0, x.share1
    y0, y1 = y.share0, y.share1

    x_shape = x.shape
    y_shape = y.shape

    z = cached_results.get(('dot',x,y), None)

    if z is None:

        if x.masked is not None:
            a, a0, a1, alpha_on_0, alpha_on_1 = x.masked
        else:

            with tf.name_scope("masking"):

                with tf.device(CRYPTO_PRODUCER):
                    a = sample(x_shape)
                    a0, a1 = share(a)

                with tf.device(SERVER_0):
                    alpha0 = crt_sub(x0, a0)

                with tf.device(SERVER_1):
                    alpha1 = crt_sub(x1, a1)

                # exchange of alphas

                with tf.device(SERVER_0):
                    alpha = reconstruct(alpha0, alpha1)
                    alpha_on_0 = alpha # needed for saving in `masked`

                with tf.device(SERVER_1):
                    alpha = reconstruct(alpha0, alpha1)
                    alpha_on_1 = alpha # needed for saving in `masked`

            x.masked = (a, a0, a1, alpha_on_0, alpha_on_1)

        if y.masked is not None:
            b, b0, b1, beta_on_0, beta_on_1 = y.masked
        else:

            with tf.name_scope("masking"):

                with tf.device(CRYPTO_PRODUCER):
                    b = sample(y_shape)
                    b0, b1 = share(b)

                with tf.device(SERVER_0):
                    beta0 = crt_sub(y0, b0)

                with tf.device(SERVER_1):
                    beta1 = crt_sub(y1, b1)

                # exchange of betas

                with tf.device(SERVER_0):
                    beta = reconstruct(beta0, beta1)
                    beta_on_0 = beta # needed for saving in `masked`

                with tf.device(SERVER_1):
                    beta = reconstruct(beta0, beta1)
                    beta_on_1 = beta # needed for saving in `masked`

            y.masked = (b, b0, b1, beta_on_0, beta_on_1)

        with tf.name_scope("dot"):

            with tf.device(CRYPTO_PRODUCER):
                ab = crt_dot(a, b)
                ab0, ab1 = share(ab)

            with tf.device(SERVER_0):
                alpha = alpha_on_0
                beta = beta_on_0
                z0 = crt_add(ab0,
                     crt_add(crt_dot(a0, beta),
                     crt_add(crt_dot(alpha, b0),
                             crt_dot(alpha, beta))))

            with tf.device(SERVER_1):
                alpha = alpha_on_1
                beta = beta_on_1
                z1 = crt_add(ab1,
                     crt_add(crt_dot(a1, beta),
                             crt_dot(alpha, b1)))

        z = PrivateVariable(z0, z1)
        z = truncate(z)
        cached_results[('dot',x,y)] = z

    return z

def gen_truncate():
    assert gcd(K, M) == 1

    # precomputation for truncation
    K_inv = decompose(inverse(K, M))
    M_wrapped = decompose(M)

    def raw_truncate(x):
        y = crt_sub(x, crt_mod(x))
        return crt_mul(y, K_inv)

    def truncate(x):
        assert isinstance(x, PrivateVariable)

        x0, x1 = x.share0, x.share1

        with tf.name_scope("truncate"):
    
            with tf.device(SERVER_0):
                y0 = raw_truncate(x0)

            with tf.device(SERVER_1):
                y1 = crt_sub(M_wrapped, raw_truncate(crt_sub(M_wrapped, x1)))

        return PrivateVariable(y0, y1)

    return truncate

truncate = gen_truncate()

class PrivateVariable:
    
    def __init__(self, share0, share1):
        self.share0 = share0
        self.share1 = share1
        self.masked = None
    
    @property
    def shape(self):
        return self.share0[0].shape

    def __add__(x, y):
        return add(x, y)
    
    def __sub__(x, y):
        return sub(x, y)
    
    def __mul__(x, y):
        return mul(x, y)

    def dot(x, y):
        return dot(x, y)

    def truncate(x):
        return truncate(x)

def define_input(shape):
    
    with tf.name_scope("input"):
        
        with tf.device(INPUT_PROVIDER):
            input_x = [ tf.placeholder(INT_TYPE, shape=shape) for _ in range(len(m)) ]
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