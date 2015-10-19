# encoding: utf-8
# Copyright (c) 2015 Grzegorz Chrupała
# Some code adapted from https://github.com/IndicoDataSolutions/Passage
# Copyright (c) 2015 IndicoDataSolutions

import theano
import theano.tensor as T
from util import *
import context
import numpy
from theano.sandbox.rng_mrg import MRG_RandomStreams

rstream = MRG_RandomStreams(seed=np.random.randint(10e6))

class Layer(object):
    """Neural net layer. Maps (a number of) theano tensors to a theano tensor."""
    def __init__(self):
        self.params = []
        self.names  = []

    def __call__(self, *inp):
         raise NotImplementedError

    def compose(self, l2):
        """Compose itself with another layer."""
        return ComposedLayer(self, l2)

    def param(self, name):
        """Return parameter by name."""
        matches = [ p for (n,p) in zip(self.names, self.params) if n == name ]
        if len(matches) == 0:
            raise KeyError(name)
        else:
            return matches[0]

class Identity(Layer):
    """Return the input unmodified."""
    def __call__(self, inp):
        return inp
    
class ComposedLayer(Layer):
    
    def __init__(self, first, second):
        autoassign(locals())
        self.params = self.first.params + self.second.params
        self.names  = self.first.names  + self.second.names

    def __call__(self, inp):
        return self.first(self.second(inp))

class Embedding(Layer):
    """Embedding (lookup table) layer."""
    def __init__(self, size_in, size_out):
        autoassign(locals())
        self.E = uniform((self.size_in, self.size_out))
        self.params = [self.E]
        self.names = ['E']

    def __call__(self, inp):
        return self.E[inp]

    def unembed(self, inp):
        """Invert the embedding."""
        return T.dot(inp, self.E.T)
        
def theano_one_hot(idx, n):
    z = T.zeros((idx.shape[0], n))
    one_hot = T.set_subtensor(z[T.arange(idx.shape[0]), idx], 1)
    return one_hot        

class OneHot(Layer):
    """One-hot encoding of input."""
    def __init__(self, size_in):
        autoassign(locals())
        self.params = []
        self.names =  []

    def __call__(self, inp):
        return theano_one_hot(inp.flatten(), self.size_in).reshape((inp.shape[0], inp.shape[1], self.size_in))
                

class Dense(Layer):
    """Fully connected layer."""
    def __init__(self, size_in, size_out):
        autoassign(locals())
        self.w = orthogonal((self.size_in, self.size_out))
        self.b = shared0s((self.size_out))
        self.params = [self.w, self.b]
        self.names  = ['w', 'b']

    def __call__(self, inp):
        return T.dot(inp, self.w) + self.b


class Dropout(Layer):
    """Randomly set `prob` fraction of input units to zero during training."""
    def __init__(self, prob):
        autoassign(locals())
        self.params = []
        self.names = []

    def __call__(self, inp):
        if self.prob > 0.0:
            keep = 1.0 - self.prob
            if context.training:
                return inp * rstream.binomial(inp.shape, p=keep, dtype=theano.config.floatX)
            else:
                return inp * keep
        else:
            return inp

class GRU_gate_activations(Layer):
    """Gated Recurrent Unit layer. Takes initial hidden state, and a
       sequence of inputs, and returns the sequence of hidden states,
       and the sequences of gate activations.
    """
    def __init__(self, size_in, size, activation=tanh, gate_activation=steeper_sigmoid):
        autoassign(locals())

        self.init = orthogonal
        self.w_z = self.init((self.size_in, self.size))
        self.w_r = self.init((self.size_in, self.size))

        self.u_z = self.init((self.size, self.size))
        self.u_r = self.init((self.size, self.size))

        self.b_z = shared0s((self.size))
        self.b_r = shared0s((self.size))

        self.w_h = self.init((self.size_in, self.size)) 
        self.u_h = self.init((self.size, self.size))
        self.b_h = shared0s((self.size))   

        self.params = [self.w_z, self.w_r, self.w_h, self.u_z, self.u_r, self.u_h, self.b_z, self.b_r, self.b_h]
        self.names =  ['w_z',    'w_r',    'w_h',    'u_z',    'u_r',    'u_h',    'b_z',    'b_r',    'b_h']

    def step(self, xz_t, xr_t, xh_t, h_tm1, u_z, u_r, u_h):
        z = self.gate_activation(xz_t + T.dot(h_tm1, u_z))
        r = self.gate_activation(xr_t + T.dot(h_tm1, u_r))
        h_tilda_t = self.activation(xh_t + T.dot(r * h_tm1, u_h))
        h_t = z * h_tm1 + (1 - z) * h_tilda_t
        return h_t, r, z

    def __call__(self, h0, seq, repeat_h0=0):
        X = seq.dimshuffle((1,0,2))
        H0 = T.repeat(h0, X.shape[1], axis=0) if repeat_h0 else h0
        x_z = T.dot(X, self.w_z) + self.b_z
        x_r = T.dot(X, self.w_r) + self.b_r
        x_h = T.dot(X, self.w_h) + self.b_h
        out, _ = theano.scan(self.step, 
            sequences=[x_z, x_r, x_h], 
                             outputs_info=[H0, None, None], 
            non_sequences=[self.u_z, self.u_r, self.u_h]
        )
        return (out[0].dimshuffle((1,0,2)), out[1].dimshuffle((1,0,2)), out[2].dimshuffle((1,0,2)))

class GRU(Layer):
    """Gated Recurrent Unit layer. Takes initial hidden state, and a
       sequence of inputs, and returns the sequence of hidden states.
    """
    def __init__(self, size_in, size, activation=tanh, gate_activation=steeper_sigmoid):
        autoassign(locals())
        self.gru = GRU_gate_activations(self.size_in, self.size, activation=self.activation,
                                        gate_activation=self.gate_activation)
        self.params = self.gru.params
        self.names  = self.gru.names

    def __call__(self, h0, seq, repeat_h0=1):
        H, _Z, _R = self.gru(h0, seq, repeat_h0=repeat_h0)
        return H
        
        
class Zeros(Layer):
    """Returns a shared variable vector of specified size initialized with zeros.""" 
    def __init__(self, size):
        autoassign(locals())
        self.zeros = theano.shared(numpy.asarray(numpy.zeros((1,self.size)), dtype=theano.config.floatX))
        self.params = [self.zeros]
        self.names = ['zeros']
    
    def __call__(self):
        return self.zeros

class WithH0(Layer):
    """Returns a new Layer which composes 'h0' and 'layer' such that 'h0()' is the initial state of 'layer'."""
    def __init__(self, h0, layer):
        autoassign(locals())
        self.params = self.h0.params + self.layer.params
        self.names  = self.h0.names + self.layer.names

    def __call__(self, inp):
        return self.layer(self.h0(), inp, repeat_h0=1)

def GRUH0(size_in, size, layer=GRU,**kwargs):
    """A GRU layer with its own initial state."""
    return WithH0(Zeros(size), layer(size_in, size, **kwargs))

class WithDropout(Layer):
    """Composes given layer with a dropout layer."""
    def __init__(self, layer, prob):
        autoassign(locals())
        self.Dropout = Dropout(prob=prob)
        self.params = params(self.layer, self.Dropout)
        self.names = names(self.layer, self.Dropout)

    def __call__(self, *args, **kwargs):
        return self.Dropout(self.layer(*args, **kwargs))


def last(x):
    """Returns the last time step of all sequences in x."""
    try:
        return x.dimshuffle((1,0,2))[-1]
    except:
        return x[0].dimshuffle((1,0,2))[-1]
    
class EncoderDecoderGRU(Layer):
    """A pair of GRUs: the first one encodes the input sequence into a
       state, the second one decodes the state into a sequence of states.
   
    Args:
      inp (tensor3) - input sequence
      out_prev (tensor3) - sequence of output elements at position -1
   
    Returns:
      tensor3 - sequence of states (one for each element of output sequence)
    """
    def __init__(self, size_in, size, size_out, encoder=GRUH0, decoder=GRU):
        self.size_in  = size_in
        self.size     = size
        self.size_out = size_out
        self.Encode   = encoder(size_in=self.size_in, size=self.size)
        self.Decode   = decoder(size_in=self.size_out, size=self.size)
        self.params = self.Encode.params + self.Decode.params
        self.names  = self.Encode.names  + self.Decode.names

    def __call__(self, inp, out_prev):
        return self.Decode(last(self.Encode(inp)), out_prev)    

                 
class StackedGRU(Layer):
    """A stack of GRUs."""
    def __init__(self, size_in, size, depth=2, dropout_prob=0.0, **kwargs):
        autoassign(locals())
        self.bottom = WithDropout(GRU(self.size_in, self.size, **kwargs), prob=self.dropout_prob)
        layers = [ WithDropout(GRUH0(self.size, self.size, **kwargs), prob=self.dropout_prob)
                   for _ in range(1,self.depth) ]
        self.stack = reduce(lambda z, x: x.compose(z), layers, Identity())
        self.params = params(self.stack, self.bottom)
        self.names  = names(self.stack, self.bottom)

    def __call__(self, h0, inp, repeat_h0=0):
        return self.stack(self.bottom(h0, inp, repeat_h0=repeat_h0))
        
def StackedGRUH0(size_in, size, depth, **kwargs): 
    """A stacked GRU layer with its own initial state."""
    return WithH0(Zeros(size), StackedGRU(size_in, size, depth, **kwargs))
