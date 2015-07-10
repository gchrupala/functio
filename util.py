# encoding: utf-8
# Copyright (c) 2015 Grzegorz Chrupała
# Some code adapted from https://github.com/IndicoDataSolutions/Passage
# Copyright (c) 2015 IndicoDataSolutions

import theano
import theano.tensor as T
import numpy as np

class IdTable(object):
    """Map hashable objects to ints and vice versa."""
    def __init__(self):
        self.encoder = {}
        self.decoder = {}
        self.max = 0

    def to_id(self, s, default=None):
        i = self.encoder.get(s, default)
        if i is not None:
            return i
        else:
            i = self.max
            self.encoder[s] = i
            self.decoder[i] = s
            self.max += 1
            return i

    def from_id(self, i):
        return self.decoder[i]


class IdMapper(object):
    """Map lists of words to lists of ints."""
    def __init__(self, min_df=1):
        self.min_df = min_df
        self.freq = {}
        self.ids = IdTable()
        self.BEG = '<BEG>'
        self.END = '<END>'
        self.UNK = '<UNK>'
        self.BEG_ID = self.ids.to_id(self.BEG)
        self.END_ID = self.ids.to_id(self.END)
        self.UNK_ID = self.ids.to_id(self.UNK)
    
    def size(self):
        return len(self.ids.encoder)

    def fit(self, sents):
        """Prepare model by collecting counts from data."""
        sents = list(sents)
        for sent in sents:
            for word in set(sent):
                self.freq[word] = self.freq.get(word, 0) + 1

    def fit_transform(self, sents):
        """Map each word in sents to a unique int, adding new words."""
        sents = list(sents)
        self.fit(sents)
        return self._transform(sents, update=True)

    def transform(self, sents):
        """Map each word in sents to a unique int, without adding new words."""
        return self._transform(sents, update=False)
            
    def _transform(self, sents, update=False):
        default = None if update else self.UNK_ID
        for sent in sents:
            ids = []
            for word in sent:
                if self.freq.get(word, 0) < self.min_df:
                    ids.append(self.UNK_ID)
                else:
                    ids.append(self.ids.to_id(word, default=default))
            yield ids
        
    def inverse_transform(self, sents):
        """Map each id in sents to the corresponding word."""
        for sent in sents:
            yield [ self.ids.from_id(i) for i in sent ]


def shared0s(shape, dtype=theano.config.floatX, name=None):
    return sharedX(np.zeros(shape), dtype=dtype, name=name)

def sharedX(X, dtype=theano.config.floatX, name=None):
    return theano.shared(np.asarray(X, dtype=dtype), name=name)

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def uniform(shape, scale=0.05):
    return sharedX(np.random.uniform(low=-scale, high=scale, size=shape))

def orthogonal(shape, scale=1.1):
    """ benanne lasagne ortho init (faster than qr approach)"""
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v # pick the one with the correct shape
    q = q.reshape(shape)
    return sharedX(scale * q[:shape[0], :shape[1]])

def tanh(x):
	return T.tanh(x)

def rectify(x):
	return (x + abs(x)) / 2.0

def clipped_rectify(x):
	return T.clip((x + abs(x)) / 2.0, 0., 5.)

def steeper_sigmoid(x):
	return 1./(1. + T.exp(-3.75 * x))

def sigmoid(x):
	return T.nnet.sigmoid(x)

def softmax3d(inp): 
    x = inp.reshape((inp.shape[0]*inp.shape[1],inp.shape[2]))
    e_x = T.exp(x - x.max(axis=1).dimshuffle(0, 'x'))
    result = e_x / e_x.sum(axis=1).dimshuffle(0, 'x')
    return result.reshape(inp.shape)

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def CrossEntropy(y_true, y_pred):
    return T.nnet.categorical_crossentropy(T.clip(y_pred, 1e-7, 1.0-1e-7), y_true).mean()

def MeanSquaredError(y_true, y_pred):
    return T.sqr(y_pred - y_true).mean()

class Adam(object):
    """Adam: a Method for Stochastic Optimization, Kingma and Ba. http://arxiv.org/abs/1412.6980."""

    def __init__(self, lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.e = e

    def get_updates(self, params, cost):
        updates = []
        grads = T.grad(cost, params)
        i = theano.shared(floatX(0.))
        i_t = i + 1.
        fix1 = 1. - self.b1**(i_t)
        fix2 = 1. - self.b2**(i_t)
        lr_t = self.lr * (T.sqrt(fix2) / fix1)
        for p, g in zip(params, grads):
            m = theano.shared(p.get_value() * 0.)
            v = theano.shared(p.get_value() * 0.)
            m_t = (self.b1 * g) + ((1. - self.b1) * m)
            v_t = (self.b2 * T.sqr(g)) + ((1. - self.b2) * v)
            g_t = m_t / (T.sqrt(v_t) + self.e)
            p_t = p - (lr_t * g_t)
            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((p, p_t))
        updates.append((i, i_t))
        return updates

def autoassign(locs):
    """Assign locals to self."""
    for key in locs.keys():
        if key!="self":
            locs["self"].__dict__[key]=locs[key]

def params(*layers):
    return sum([ layer.params for layer in layers ], [])
