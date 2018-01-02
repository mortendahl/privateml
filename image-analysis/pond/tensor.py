import random
import numpy as np


class NativeTensor:
    
    def __init__(self, values):
        self.values = values
        
    def from_values(values):
        return NativeTensor(values)
    
    @property
    def size(self):
        return self.values.size
    
    @property
    def shape(self):
        return self.values.shape
    
    def __getitem__(self, index):
        return NativeTensor(self.values[index])
    
    def concatenate(self, other):
        assert isinstance(other, NativeTensor), type(other)
        return NativeTensor.from_values(np.concatenate([self.values, other.values]))

    def reveal(self):
        return self
    
    def unwrap(self):
        return self.values
    
    def __repr__(self):
        return "NativeTensor(%s)" % self.values
    
    def wrap_if_needed(y):
        if isinstance(y, int) or isinstance(y, float): return NativeTensor.from_values(np.array([y]))
        if isinstance(y, np.ndarray): return NativeTensor.from_values(y)
        return y
    
    def add(x, y):
        y = NativeTensor.wrap_if_needed(y)
        if isinstance(y, NativeTensor): return NativeTensor(x.values + y.values)
        if isinstance(y, PublicEncodedTensor): return PublicEncodedTensor.from_values(x.values).add(y)
        if isinstance(y, PrivateEncodedTensor): return PublicEncodedTensor.from_values(x.values).add(y)
        raise TypeError("%s does not support %s" % (type(x), type(y)))
        
    def __add__(x, y):
        return x.add(y)
    
    def sub(x, y):
        y = NativeTensor.wrap_if_needed(y)
        if isinstance(y, NativeTensor): return NativeTensor(x.values - y.values)
        if isinstance(y, PublicEncodedTensor): return PublicEncodedTensor.from_values(x.values).sub(y)
        if isinstance(y, PrivateEncodedTensor): return PublicEncodedTensor.from_values(x.values).sub(y)
        raise TypeError("%s does not support %s" % (type(x), type(y)))
        
    def __sub__(x, y):
        return x.sub(y)
    
    def mul(x, y):
        y = NativeTensor.wrap_if_needed(y)
        if isinstance(y, NativeTensor): return NativeTensor(x.values * y.values)
        if isinstance(y, PublicEncodedTensor): return PublicEncodedTensor.from_values(x.values).mul(y)
        if isinstance(y, PrivateEncodedTensor): return PublicEncodedTensor.from_values(x.values).mul(y)
        raise TypeError("%s does not support %s" % (type(x), type(y)))
        
    def __mul__(x, y):
        return x.mul(y)
        
    def dot(x, y):
        y = NativeTensor.wrap_if_needed(y)
        if isinstance(y, NativeTensor): return NativeTensor(x.values.dot(y.values))
        if isinstance(y, PublicEncodedTensor): return PublicEncodedTensor.from_values(x.values).dot(y)
        if isinstance(y, PrivateEncodedTensor): return PublicEncodedTensor.from_values(x.values).dot(y)
        raise TypeError("%s does not support %s" % (type(x), type(y)))
        
    def div(x, y):
        y = NativeTensor.wrap_if_needed(y)
        if isinstance(y, NativeTensor): return NativeTensor(x.values / y.values)
        raise TypeError("%s does not support %s" % (type(x), type(y)))
        
    def __div__(x, y):
        return x.div(y)
        
    def transpose(x):
        return NativeTensor(x.values.T)
    
    def neg(x):
        return NativeTensor(0 - x.values)

    def sum(x, axis):
        return NativeTensor(x.values.sum(axis=axis, keepdims=True))
    
    def argmax(x, axis):
        return NativeTensor.from_values(x.values.argmax(axis=axis))
    
    def exp(x):
        return NativeTensor(np.exp(x.values))
    
    def log(x):
        return NativeTensor(np.log(x.values))
    
    def inv(x):
        return NativeTensor(1. / x.values)
    

DTYPE = 'object'
Q = 2657003489534545107915232808830590043


from math import log
log2 = lambda x: log(x)/log(2)

# for arbitrary precision ints

# we need room for summing MAX_SUM values of MAX_DEGREE before during modulus reduction
MAX_DEGREE = 2
MAX_SUM = 2**12
assert MAX_DEGREE * log2(Q) + log2(MAX_SUM) < 256

BASE = 2
PRECISION_INTEGRAL   = 16
PRECISION_FRACTIONAL = 32
# TODO Gap as needed for local truncating

# we need room for double precision before truncating
assert PRECISION_INTEGRAL + 2 * PRECISION_FRACTIONAL < log(Q)/log(BASE)

def encode(rationals):
    return (rationals * BASE**PRECISION_FRACTIONAL).astype('int').astype(DTYPE) % Q

def decode(elements):
    map_negative_range = np.vectorize(lambda element: element if element <= Q/2 else element - Q)
    return map_negative_range(elements) / BASE**PRECISION_FRACTIONAL


def wrap_if_needed(y):
    if isinstance(y, int) or isinstance(y, float): return PublicEncodedTensor.from_values(np.array([y]))
    if isinstance(y, np.ndarray): return PublicEncodedTensor.from_values(y)
    if isinstance(y, NativeTensor): return PublicEncodedTensor.from_values(y.values)
    return y

class PublicEncodedTensor:
    
    def __init__(self, values, elements=None):
        if not values is None:
            if not isinstance(values, np.ndarray):
                values = np.array([values])
            elements = encode(values)
        assert isinstance(elements, np.ndarray), "%s, %s, %s" % (values, elements, type(elements))
        self.elements = elements
    
    def from_values(values):
        return PublicEncodedTensor(values)
    
    def from_elements(elements):
        return PublicEncodedTensor(None, elements)
    
    def __repr__(self):
        return "PublicEncodedTensor(%s)" % decode(self.elements)
    
    def __getitem__(self, index):
        return PublicEncodedTensor.from_elements(self.elements[index])
    
    def concatenate(self, other):
        assert isinstance(other, PublicEncodedTensor), type(other)
        return PublicEncodedTensor.from_elements(np.concatenate([self.elements, other.elements]))
    
    @property
    def shape(self):
        return self.elements.shape
    
    @property
    def size(self):
        return self.elements.size
    
    def unwrap(self):
        return decode(self.elements)

    def reveal(self):
        return NativeTensor.from_values(decode(self.elements))

    def truncate(self, amount=PRECISION_FRACTIONAL):
        positive_numbers = (self.elements <= Q // 2).astype(int)
        elements = self.elements
        elements = (Q + (2 * positive_numbers - 1) * elements) % Q # x if x <= Q//2 else Q - x
        elements = np.floor_divide(elements, BASE**amount)         # x // BASE**amount
        elements = (Q + (2 * positive_numbers - 1) * elements) % Q # x if x <= Q//2 else Q - x
        return PublicEncodedTensor.from_elements(elements)
    
    def add(x, y):
        y = wrap_if_needed(y)
        if isinstance(y, PublicEncodedTensor): 
            return PublicEncodedTensor.from_elements((x.elements + y.elements) % Q)
        if isinstance(y, PrivateEncodedTensor):
            shares0 = (x.elements + y.shares0) % Q
            shares1 =               y.shares1
            return PrivateEncodedTensor.from_shares(shares0, shares1)
        raise TypeError("%s does not support %s" % (type(x), type(y)))
        
    def __add__(x, y):
        return x.add(y)
        
    def sub(x, y):
        y = wrap_if_needed(y)
        if isinstance(y, PublicEncodedTensor): return PublicEncodedTensor.from_elements((x.elements - y.elements) % Q)
        if isinstance(y, PrivateEncodedTensor): return x.add(y.neg()) # TODO there might be a more efficient way
        raise TypeError("%s does not support %s" % (type(x), type(y)))
        
    def __sub__(x, y):
        return x.sub(y)
        
    def mul(x, y):
        y = wrap_if_needed(y)
        if isinstance(y, PublicFieldTensor): return PublicFieldTensor.from_elements((x.elements * y.elements) % Q)
        if isinstance(y, PublicEncodedTensor): return PublicEncodedTensor.from_elements((x.elements * y.elements) % Q).truncate()
        if isinstance(y, PrivateEncodedTensor):
            shares0 = (x.elements * y.shares0) % Q
            shares1 = (x.elements * y.shares1) % Q
            return PrivateEncodedTensor.from_shares(shares0, shares1).truncate()
        raise TypeError("%s does not support %s" % (type(x), type(y)))
    
    def __mul__(x, y):
        return x.mul(y)
    
    def dot(x, y):
        y = wrap_if_needed(y)
        if isinstance(y, PublicEncodedTensor): return PublicEncodedTensor.from_elements(x.elements.dot(y.elements) % Q).truncate()
        if isinstance(y, PrivateEncodedTensor):
            shares0 = x.elements.dot(y.shares0) % Q
            shares1 = x.elements.dot(y.shares1) % Q
            return PrivateEncodedTensor.from_shares(shares0, shares1).truncate()
        raise TypeError("%s does not support %s" % (type(x), type(y)))
    
    def div(x, y):
        y = wrap_if_needed(y)
        if isinstance(y, NativeTensor): return x.mul(y.inv())
        if isinstance(y, PublicEncodedTensor): return x.mul(y.inv())
        raise TypeError("%s does not support %s" % (type(x), type(y)))
    
    def __div__(x, y):
        return x.div(y)
        
    def transpose(x):
        return PublicEncodedTensor.from_elements(x.elements.T)
    
    def sum(x, axis):
        return PublicEncodedTensor.from_elements(x.elements.sum(axis=axis, keepdims=True))
    
    def argmax(x, axis):
        return PublicEncodedTensor.from_values(decode(x.elements).argmax(axis=axis))
    
    def neg(x):
        return PublicEncodedTensor.from_values(decode(x.elements) * -1)
    
    def inv(x):
        return PublicEncodedTensor.from_values(1. / decode(x.elements))


class PublicFieldTensor:
    
    def __init__(self, elements):
        self.elements = elements

    def from_elements(elements):
        return PublicFieldTensor(elements)
        
    def __repr__(self):
        return "PublicFieldTensor(%s)" % self.elements

    def __getitem__(self, index):
        return PublicFieldTensor.from_elements(self.elements[index])

    @property
    def size(self):
        return self.elements.size
    
    @property
    def shape(self):
        return self.elements.shape
    
    def add(x, y):
        if isinstance(y, PublicFieldTensor): 
            return PublicFieldTensor.from_elements((x.elements + y.elements) % Q)
        if isinstance(y, PrivateFieldTensor):
            shares0 = (x.elements + y.shares0) % Q
            shares1 =               y.shares1
            return PrivateFieldTensor.from_shares(shares0, shares1)
        raise TypeError("%s does not support %s" % (type(x), type(y)))
        
    def __add__(x, y):
        return x.add(y)
    
    def mul(x, y):
        if isinstance(y, PublicFieldTensor): 
            return PublicFieldTensor.from_elements((x.elements * y.elements) % Q)
        if isinstance(y, PrivateFieldTensor):
            shares0 = (x.elements * y.shares0) % Q
            shares1 = (x.elements * y.shares1) % Q
            return PrivateFieldTensor.from_shares(shares0, shares1)
        raise TypeError("%s does not support %s" % (type(x), type(y)))
    
    def __mul__(x, y):
        return x.mul(y)
    
    def dot(x, y):
        if isinstance(y, PublicFieldTensor): 
            return PublicFieldTensor.from_elements((x.elements.dot(y.elements)) % Q)
        if isinstance(y, PrivateFieldTensor):
            shares0 = (x.elements.dot(y.shares0)) % Q
            shares1 = (x.elements.dot(y.shares1)) % Q
            return PrivateFieldTensor.from_shares(shares0, shares1)
        raise TypeError("%s does not support %s" % (type(x), type(y)))


def share(elements):
    shares0 = np.array([ random.randrange(Q) for _ in range(elements.size) ]).astype(DTYPE).reshape(elements.shape)
    shares1 = (elements - shares0) % Q
    return shares0, shares1

def reconstruct(shares0, shares1):
    return (shares0 + shares1) % Q


class PrivateFieldTensor:
    
    def __init__(self, elements, shares0=None, shares1=None):
        if not elements is None:
            shares0, shares1 = share(elements)
        assert isinstance(shares0, np.ndarray), "%s, %s, %s" % (values, shares0, type(shares0))
        assert isinstance(shares1, np.ndarray), "%s, %s, %s" % (values, shares1, type(shares1))
        assert shares0.shape == shares1.shape
        self.shares0 = shares0
        self.shares1 = shares1
    
    def from_elements(elements):
        return PrivateFieldTensor(elements)
    
    def from_shares(shares0, shares1):
        return PrivateFieldTensor(None, shares0, shares1)
    
    def reveal(self):
        return PublicFieldTensor.from_elements(reconstruct(self.shares0, self.shares1))
    
    def __repr__(self):
        return "PrivateFieldTensor(%s)" % self.reveal().elements

    def __getitem__(self, index):
        return PrivateFieldTensor.from_shares(self.shares0[index], self.shares1[index])
    
    @property
    def size(self):
        return self.shares0.size
    
    @property
    def shape(self):
        return self.shares0.shape
    
    def add(x, y):
        if isinstance(y, PrivateFieldTensor):
            shares0 = (x.shares0 + y.shares0) % Q
            shares1 = (x.shares1 + y.shares1) % Q
            return PrivateFieldTensor.from_shares(shares0, shares1)
        if isinstance(y, PublicFieldTensor):
            shares0 = (x.shares0 + y.elements) % Q
            shares1 =  x.shares1
            return PrivateFieldTensor.from_shares(shares0, shares1)
        raise TypeError("%s does not support %s" % (type(x), type(y)))
        
    def __add__(x, y):
        return x.add(y)
    
    def mul(x, y):
        if isinstance(y, PublicFieldTensor):
            shares0 = (x.shares0 * y.elements) % Q
            shares1 = (x.shares1 * y.elements) % Q
            return PrivateFieldTensor.from_shares(shares0, shares1)
        raise TypeError("%s does not support %s" % (type(x), type(y)))
    
    def __mul__(x, y):
        return x.mul(y)
    
    def dot(x, y):
        if isinstance(y, PublicFieldTensor):
            shares0 = (x.shares0.dot(y.elements)) % Q
            shares1 = (x.shares1.dot(y.elements)) % Q
            return PrivateFieldTensor.from_shares(shares0, shares1)
        raise TypeError("%s does not support %s" % (type(x), type(y)))


def generate_mul_triple(shape):
    a = np.array([ random.randrange(Q) for _ in range(np.prod(shape)) ]).astype(DTYPE).reshape(shape)
    b = np.array([ random.randrange(Q) for _ in range(np.prod(shape)) ]).astype(DTYPE).reshape(shape)
    ab = (a * b) % Q
    return PrivateFieldTensor.from_elements(a), \
           PrivateFieldTensor.from_elements(b), \
           PrivateFieldTensor.from_elements(ab)

def generate_dot_triple(m, n, o):
    a = np.array([ random.randrange(Q) for _ in range(m*n) ]).astype(DTYPE).reshape((m,n))
    b = np.array([ random.randrange(Q) for _ in range(n*o) ]).astype(DTYPE).reshape((n,o))
    ab = np.dot(a, b)
    return PrivateFieldTensor.from_elements(a), \
           PrivateFieldTensor.from_elements(b), \
           PrivateFieldTensor.from_elements(ab)

# def generate_mul_triple(shape):
#     a = np.zeros(shape).astype(int).astype(DTYPE)
#     b = np.zeros(shape).astype(int).astype(DTYPE)
#     ab = (a * b) % Q
#     return PrivateFieldTensor.from_elements(a), \
#            PrivateFieldTensor.from_elements(b), \
#            PrivateFieldTensor.from_elements(ab)
# 
# def generate_dot_triple(m, n, o):
#     a = np.zeros((m,n)).astype(int).astype(DTYPE)
#     b = np.zeros((n,o)).astype(int).astype(DTYPE)
#     ab = np.dot(a, b)
#     return PrivateFieldTensor.from_elements(a), \
#            PrivateFieldTensor.from_elements(b), \
#            PrivateFieldTensor.from_elements(ab)
           
           
class PrivateEncodedTensor:
    
    def __init__(self, values, shares0=None, shares1=None):
        if not values is None:
            if not isinstance(values, np.ndarray):
                values = np.array([values])
            shares0, shares1 = share(encode(values))
        assert isinstance(shares0, np.ndarray), "%s, %s, %s" % (values, shares0, type(shares0))
        assert isinstance(shares1, np.ndarray), "%s, %s, %s" % (values, shares1, type(shares1))
        assert shares0.shape == shares1.shape
        self.shares0 = shares0
        self.shares1 = shares1
    
    def from_values(values):
        return PrivateEncodedTensor(values)
    
    def from_elements(elements):
        shares0, shares1 = share(elements)
        return PrivateEncodedTensor(None, shares0, shares1)
    
    def from_shares(shares0, shares1):
        return PrivateEncodedTensor(None, shares0, shares1)
    
    def __repr__(self):
        elements = (self.shares0 + self.shares1) % Q
        return "PrivateEncodedTensor(%s)" % decode(elements)
    
    def __getitem__(self, index):
        return PrivateEncodedTensor.from_shares(self.shares0[index], self.shares1[index])
    
    def concatenate(self, other):
        assert isinstance(other, PrivateEncodedTensor), type(other)
        shares0 = np.concatenate([self.shares0, other.shares0])
        shares1 = np.concatenate([self.shares1, other.shares1])
        return PrivateEncodedTensor.from_shares(shares0, shares1)
    
    @property
    def shape(self):
        return self.shares0.shape
    
    @property
    def size(self):
        return self.shares0.size
    
    def unwrap(self):
        return decode((self.shares0 + self.shares1) % Q)
    
    def reveal(self):
        return NativeTensor.from_values(decode((self.shares0 + self.shares1) % Q))
    
    def truncate(self, amount=PRECISION_FRACTIONAL):
        shares0 = np.floor_divide(self.shares0, BASE**amount) % Q
        shares1 = (Q - (np.floor_divide(Q - self.shares1, BASE**amount))) % Q
        return PrivateEncodedTensor.from_shares(shares0, shares1)
    
    def add(x, y):
        y = wrap_if_needed(y)
        if isinstance(y, PublicEncodedTensor):
            shares0 = (x.shares0 + y.elements) % Q
            shares1 =  x.shares1
            return PrivateEncodedTensor.from_shares(shares0, shares1)
        if isinstance(y, PrivateEncodedTensor):
            shares0 = (x.shares0 + y.shares0) % Q
            shares1 = (x.shares1 + y.shares1) % Q
            return PrivateEncodedTensor.from_shares(shares0, shares1)
        raise TypeError("%s does not support %s" % (type(x), type(y)))
        
    def __add__(x, y):
        return x.add(y)
    
    def sub(x, y):
        y = wrap_if_needed(y)
        if isinstance(y, PublicEncodedTensor):
            shares0 = (x.shares0 - y.elements) % Q
            shares1 =  x.shares1
            return PrivateEncodedTensor.from_shares(shares0, shares1)
        if isinstance(y, PrivateEncodedTensor):
            shares0 = (x.shares0 - y.shares0) % Q
            shares1 = (x.shares1 - y.shares1) % Q
            return PrivateEncodedTensor.from_shares(shares0, shares1)
        if isinstance(y, PrivateFieldTensor):
            shares0 = (x.shares0 - y.shares0) % Q
            shares1 = (x.shares1 - y.shares1) % Q
            return PrivateFieldTensor.from_shares(shares0, shares1)
        raise TypeError("%s does not support %s" % (type(x), type(y)))
        
    def __sub__(x, y):
        return x.sub(y)
    
    def mul(x, y, precomputed=None):
        y = wrap_if_needed(y)
        if isinstance(y, PublicEncodedTensor):
            shares0 = (x.shares0 * y.elements) % Q
            shares1 = (x.shares1 * y.elements) % Q
            return PrivateEncodedTensor.from_shares(shares0, shares1).truncate()
        if isinstance(y, PrivateEncodedTensor):
            if precomputed is None: precomputed = generate_mul_triple(x.shape)
            a, b, ab = precomputed
            assert x.shape == y.shape
            assert x.shape == a.shape
            assert y.shape == b.shape
            alpha = (x - a).reveal()
            beta  = (y - b).reveal()
            z = alpha.mul(beta) + \
                alpha.mul(b) + \
                a.mul(beta) + \
                ab
            return PrivateEncodedTensor.from_shares(z.shares0, z.shares1).truncate()
        if isinstance(y, PrivateFieldTensor):
            shares0 = (x.shares0 * y.shares0) % Q
            shares1 = (x.shares1 * y.shares1) % Q
            return PrivateFieldTensor.from_shares(shares0, shares1)
        if isinstance(y, PublicFieldTensor):
            shares0 = (x.shares0 * y.elements) % Q
            shares1 = (x.shares1 * y.elements) % Q
            return PrivateFieldTensor.from_shares(shares0, shares1)
        raise TypeError("%s does not support %s" % (type(x), type(y)))
        
    def __mul__(x, y):
        return x.mul(y)
    
    def dot(x, y, precomputed=None):
        y = wrap_if_needed(y)
        if isinstance(y, PublicEncodedTensor):
            assert x.shape[1] == y.shape[0]
            shares0 = x.shares0.dot(y.elements) % Q
            shares1 = x.shares1.dot(y.elements) % Q
            return PrivateEncodedTensor.from_shares(shares0, shares1).truncate()
        if isinstance(y, PrivateEncodedTensor):
            m = x.shape[0]
            n = x.shape[1]
            o = y.shape[1]
            assert n == y.shape[0]
            if precomputed is None: precomputed = generate_dot_triple(m, n, o)
            a, b, ab = precomputed
            alpha = (x - a).reveal()
            beta  = (y - b).reveal()
            z = alpha.dot(beta) + \
                alpha.dot(b) + \
                a.dot(beta) + \
                ab
            return PrivateEncodedTensor.from_shares(z.shares0, z.shares1).truncate()
        raise TypeError("%s does not support %s" % (type(x), type(y)))
    
    def div(x, y):
        y = wrap_if_needed(y)
        if isinstance(y, NativeTensor): return x.mul(y.inv())
        if isinstance(y, PublicEncodedTensor): return x.mul(y.inv())
        raise TypeError("%s does not support %s" % (type(x), type(y)))
    
    def neg(self):
        minus_one = PublicFieldTensor.from_elements(np.array([Q - 1]))
        z = self.mul(minus_one)
        return PrivateEncodedTensor.from_shares(z.shares0, z.shares1)
        
    def transpose(self):
        return PrivateEncodedTensor.from_shares(self.shares0.T, self.shares1.T)
    
    def sum(self, axis):
        shares0 = self.shares0.sum(axis=axis, keepdims=True) % Q
        shares1 = self.shares1.sum(axis=axis, keepdims=True) % Q
        return PrivateEncodedTensor.from_shares(shares0, shares1)


ANALYTIC_STORE = []
NEXT_ID = 0


class AnalyticTensor:
    
    def __init__(self, values, shape=None, ident=None):
        if not values is None:
            if not isinstance(values, np.ndarray):
                values = np.array([values])
            shape = values.shape
        if ident is None:
            global NEXT_ID
            ident = "tensor_%d" % NEXT_ID
            NEXT_ID += 1
        self.shape = shape
        self.ident = ident
    
    def from_shape(shape, ident=None):
        return AnalyticTensor(None, shape, ident)
    
    def __repr__(self):
        return "AnalyticTensor(%s, %s)" % (self.shape, self.ident)
    
    def __getitem__(self, index):
        start, stop, _ = index.indices(self.shape[0])
        shape = list(self.shape)
        shape[0] = stop - start
        ident = "%s_%d,%d" % (self.ident, start, stop)
        return AnalyticTensor.from_shape(tuple(shape), ident)
    
    def reset():
        global ANALYTIC_STORE
        ANALYTIC_STORE = []
    
    def store():
        global ANALYTIC_STORE
        return ANALYTIC_STORE
    
    @property
    def size(self):
        return np.prod(self.shape)
    
    def reveal(self):
        return self
    
    def wrap_if_needed(y):
        if isinstance(y, int) or isinstance(y, float): return AnalyticTensor.from_shape((1,))
        if isinstance(y, np.ndarray): return AnalyticTensor.from_shape(y.shape)
        return y
    
    def add(x, y):
        y = AnalyticTensor.wrap_if_needed(y)
        ANALYTIC_STORE.append(('add', x, y))
        return AnalyticTensor.from_shape(x.shape)
        
    def __add__(x, y):
        return x.add(y)
    
    def sub(x, y):
        y = AnalyticTensor.wrap_if_needed(y)
        ANALYTIC_STORE.append(('sub', x, y))
        return AnalyticTensor.from_shape(x.shape)
        
    def __sub__(x, y):
        return x.sub(y)
    
    def mul(x, y):
        y = AnalyticTensor.wrap_if_needed(y)
        ANALYTIC_STORE.append(('mul', x, y))
        return AnalyticTensor.from_shape(x.shape)
        
    def __mul__(x, y):
        return x.mul(y)
    
    def dot(x, y):
        y = AnalyticTensor.wrap_if_needed(y)
        ANALYTIC_STORE.append(('dot', x, y))
        return AnalyticTensor.from_shape(x.shape)
    
    def div(x, y):
        y = AnalyticTensor.wrap_if_needed(y)
        ANALYTIC_STORE.append(('div', x, y))
        return AnalyticTensor.from_shape(x.shape)
    
    def neg(self):
        ANALYTIC_STORE.append(('neg', self))
        return AnalyticTensor.from_shape(self.shape)
        
    def transpose(self):
        ANALYTIC_STORE.append(('transpose', self))
        return self
    
    def sum(self, axis):
        ANALYTIC_STORE.append(('sum', self))
        return AnalyticTensor.from_shape(self.shape)
    