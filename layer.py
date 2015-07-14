# encoding: utf-8
# Copyright (c) 2015 Grzegorz Chrupała
# Some code adapted from https://github.com/IndicoDataSolutions/Passage
# Copyright (c) 2015 IndicoDataSolutions

import theano
import theano.tensor as T
from util import *
import numpy

class Layer(object):
    """Neural net layer. Maps (a number of) theano tensors to a theano tensor."""
    def __init__(self):
        self.tag = []
        self.params = []

    def __call__(self, *inp):
         raise NotImplementedError

    def compose(self, l2):
        """Compose itself with another layer."""
        return ComposedLayer(self, l2)

class ComposedLayer(Layer):
    
    def __init__(self, first, second):
        self.first = first
        self.second = second
        self.params = self.first.params + self.second.params

    def __call__(self, inp):
        return self.first(self.second(inp))

class Embedding(Layer):
    """Embedding (lookup table) layer."""
    def __init__(self, size_in, size_out):
        self.size_in = size_in
        self.size_out = size_out
        self.E = uniform((size_in, size_out))
        self.params = [self.E]

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
        self.size_in = size_in
        self.params = []

    def __call__(self, inp):
        return theano_one_hot(inp.flatten(), self.size_in).reshape((inp.shape[0], inp.shape[1], self.size_in))
        

class Dense(Layer):
    """Fully connected layer."""
    def __init__(self, size_in, size_out):
        self.size_in = size_in
        self.size_out = size_out
        self.w = orthogonal((self.size_in, self.size_out))
        self.b = shared0s((self.size_out))
        self.params = [self.w, self.b]

    def __call__(self, inp):
        return T.dot(inp, self.w) + self.b

class GRU(Layer):
    """Gated Recurrent Unit layer. Takes initial hidden state, and a
       sequence of inputs, and returns the sequence of hidden states.
    """
    def __init__(self, size_in, size):
        self.size_in = size_in
        self.size = size
        self.activation = tanh
        self.gate_activation = steeper_sigmoid
        self.init = orthogonal
        self.size = size

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

    def step(self, xz_t, xr_t, xh_t, h_tm1, u_z, u_r, u_h):
        z = self.gate_activation(xz_t + T.dot(h_tm1, u_z))
        r = self.gate_activation(xr_t + T.dot(h_tm1, u_r))
        h_tilda_t = self.activation(xh_t + T.dot(r * h_tm1, u_h))
        h_t = z * h_tm1 + (1 - z) * h_tilda_t
        return h_t

    def __call__(self, h0, seq, repeat_h0=0):
        X = seq.dimshuffle((1,0,2))
        H0 = T.repeat(h0, X.shape[1], axis=0) if repeat_h0 else h0
        x_z = T.dot(X, self.w_z) + self.b_z
        x_r = T.dot(X, self.w_r) + self.b_r
        x_h = T.dot(X, self.w_h) + self.b_h
        out, _ = theano.scan(self.step, 
            sequences=[x_z, x_r, x_h], 
                             outputs_info=[H0], 
            non_sequences=[self.u_z, self.u_r, self.u_h]
        )
        return out.dimshuffle((1,0,2))

    
class GRU_MUT1(Layer):
    """Implementation of MUT1 from 
    http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf
    """
    def __init__(self, size_in, size):
        self.size_in = size_in
        self.size = size
        self.activation = tanh
        self.gate_activation = steeper_sigmoid
        self.init = orthogonal
        self.size = size

        self.w_z = self.init((self.size_in, self.size))
        self.w_r = self.init((self.size_in, self.size))

        self.u_z = self.init((self.size, self.size))
        self.u_r = self.init((self.size, self.size))

        self.b_z = shared0s((self.size))
        self.b_r = shared0s((self.size))

        #self.w_h = self.init((self.size_in, self.size)) 
        self.u_h = self.init((self.size, self.size))
        self.b_h = shared0s((self.size))   

        self.params = [self.w_z, self.w_r, self.u_z, self.u_r, self.u_h, self.b_z, self.b_r, self.b_h]

    def step(self, xz_t, xr_t, x_t, h_tm1, u_z, u_r, u_h):
        z = self.gate_activation(xz_t + T.dot(h_tm1, u_z))
        r = self.gate_activation(xr_t + T.dot(h_tm1, u_r))
        h_t = self.gate_activation(T.dot(r*h_tm1, u_h) + self.activation(x_t) + self.b_h) * z + (1-z) * h_tm1 
        return h_t

    def __call__(self, h0, seq, repeat_h0=0):
        X = seq.dimshuffle((1,0,2))
        H0 = T.repeat(h0, X.shape[1], axis=0) if repeat_h0 else h0
        x_z = T.dot(X, self.w_z) + self.b_z
        x_r = T.dot(X, self.w_r) + self.b_r
        #x_h = T.dot(X, self.w_h) + self.b_h
        out, _ = theano.scan(self.step, 
            sequences=[x_z, x_r, X], 
                             outputs_info=[H0], 
            non_sequences=[self.u_z, self.u_r, self.u_h]
        )
        return out.dimshuffle((1,0,2))

    
    
class GRU_MUT2(Layer):
    """Implementation of MUT2 from 
    http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf
    """
    def __init__(self, size_in, size):
        self.size_in = size_in
        self.size = size
        self.activation = tanh
        self.gate_activation = steeper_sigmoid
        self.init = orthogonal
        self.size = size

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

    def step(self, xz_t, xr_t, xh_t, h_tm1, u_z, u_r, u_h):
        z = self.gate_activation(xz_t + T.dot(h_tm1, u_z))
        r = self.gate_activation(xr_t + T.dot(h_tm1, u_r))
        h_t = self.gate_activation(T.dot(r*h_tm1, u_h) + xh_t) * z + (1-z) * h_tm1 
        return h_t

    def __call__(self, h0, seq, repeat_h0=0):
        X = seq.dimshuffle((1,0,2))
        H0 = T.repeat(h0, X.shape[1], axis=0) if repeat_h0 else h0
        x_z = T.dot(X, self.w_z) + self.b_z
        x_r = T.dot(X, self.w_r) + self.b_r
        x_h = T.dot(X, self.w_h) + self.b_h
        out, _ = theano.scan(self.step, 
            sequences=[x_z, x_r, x_h], 
                             outputs_info=[H0], 
            non_sequences=[self.u_z, self.u_r, self.u_h]
        )
        return out.dimshuffle((1,0,2))
    
    
class GRU_akos(Layer):
    """Gated Recurrent Unit layer. Takes initial hidden state, and a
       sequence of inputs, and returns the sequence of hidden states.
       WARNING: It breaks the 'last' function
    """
    def __init__(self, size_in, size, activation=tanh, gate_activation=steeper_sigmoid,
                reverse=False):
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

    def step(self, xz_t, xr_t, xh_t, h_tm1, r_tm1, z_tm1, u_z, u_r, u_h):
        z = self.gate_activation(xz_t + T.dot(h_tm1, u_z))
        r = self.gate_activation(xr_t + T.dot(h_tm1, u_r))
        h_tilda_t = self.activation(xh_t + T.dot(r * h_tm1, u_h))
        h_t = z * h_tm1 + (1 - z) * h_tilda_t
        return h_t, r, z

    def __call__(self, h0, seq, repeat_h0=1):
        X = seq.dimshuffle((1,0,2))
        if self.reverse == True:
            X = X[::-1, :, :]
            
        H0 = T.repeat(h0, X.shape[1], axis=0) 
        R0 = T.repeat(h0, X.shape[1], axis=0)
        Z0 = T.repeat(h0, X.shape[1], axis=0)
        x_z = T.dot(X, self.w_z) + self.b_z
        x_r = T.dot(X, self.w_r) + self.b_r
        x_h = T.dot(X, self.w_h) + self.b_h
        H, _ = theano.scan(self.step, 
            sequences=[x_z, x_r, x_h], outputs_info=[H0, R0, Z0], 
            non_sequences=[self.u_z, self.u_r, self.u_h]
        )
        H[0] = H[0].dimshuffle((1,0,2))
        H[1] = H[1].dimshuffle((1,0,2))
        H[2] = H[2].dimshuffle((1,0,2))
        
        return H
        
class Zeros(Layer):
    """Returns a shared variable vector of specified size initialized with zeros.""" 
    def __init__(self, size):
        self.size  = size
        self.zeros = theano.shared(numpy.asarray(numpy.zeros((1,self.size)), dtype=theano.config.floatX))
        self.params = [self.zeros]
    
    def __call__(self):
        return self.zeros

class WithH0(Layer):
    """Returns a new Layer which composes 'h0' and 'layer' such that 'h0()' is the initial state of 'layer'."""
    def __init__(self, h0, layer):
        self.h0 = h0
        self.layer = layer
        self.params = self.h0.params + self.layer.params

    def __call__(self, inp):
        return self.layer(self.h0(), inp, repeat_h0=1)

def GRUH0(size_in, size, layer=GRU,**kwargs):
    """A GRU layer with its own initial state."""
    return WithH0(Zeros(size), layer(size_in, size, **kwargs))
    
def last(x):
    """Returns the last time step of all sequences in x."""
    return x.dimshuffle((1,0,2))[-1]

def lastb(x):
    """THIS WORKS WITH GRU_akos"""
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

    def __call__(self, inp, out_prev):
        return self.Decode(last(self.Encode(inp)), out_prev)    

class StackedGRU(Layer):
    """A stack of GRUs."""
    def __init__(self, size_in, size, depth=2):
        self.size_in = size_in
        self.size = size
        self.depth = depth
        self.bottom = GRU(self.size_in, self.size)
        layers = [ GRUH0(self.size, self.size)
                   for _ in range(1,self.depth) ]
        self.stack = reduce(lambda z, x: z.compose(x), layers)
        self.params = self.stack.params

    def __call__(self, h0, inp, repeat_h0=0):
        return self.stack(self.bottom(h0, inp, repeat_h0=repeat_h0))
        
def StackedGRUH0(size_in, size, depth):
    """A stacked GRU layer with its own initial state."""
    return WithH0(Zeros(size), StackedGRU(size_in, size, depth))

