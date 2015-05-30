import theano
import theano.tensor as T
from util import *
import numpy

class Layer(object):
    """Neural net layer. Maps (a number of) theano tensors to a theano tensor."""
    def __init__(self):
        self.params = []

    def __call__(self, *inp):
        raise NotImplementedError

    def compose(self, l2):
        """Compose itself with another layer."""
        l = Layer()
        l.__call__ = lambda inp: self(l2(inp))
        l.params = self.params + l2.params
        return l


class Embedding(Layer):
    """Embedding (lookup table) layer."""
    def __init__(self, size_in, size_out):
        self.size_in = size_in
        self.size_out = size_out
        self.E = uniform((size_in, size_out))
        self.params = [self.E]

    def __call__(self, inp):
        return self.E[inp]

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

class GRU(object):
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
        
class Zeros(Layer):
    """Returns a shared variable vector of specified size initialized with zeros.""" 
    def __init__(self, size):
        self.size  = size
        self.zeros = theano.shared(numpy.asarray(numpy.zeros((1,self.size)), dtype=theano.config.floatX))
        self.params = [self.zeros]
    
    def __call__(self):
        return self.zeros

def last(x):
    """Returns the last time step of all sequences in x."""
    return x.dimshuffle((1,0,2))[-1]
    
class EncoderDecoderGRU(object):
    """A pair of GRUs: the first one encodes the input sequence into a
       state, the second one decodes the state into a sequence of states.
   
    Args:
      inp (tensor3) - input sequence
      out_prev (tensor3) - sequence of output elements at position -1
   
    Returns:
      tensor3 - sequence of states (one for each element of output sequence)
    """
    def __init__(self, size_in, size, size_out):
        self.size_in  = size_in
        self.size     = size
        self.size_out = size_out
        self.Encode   = GRU(size_in=self.size_in, size=self.size)
        self.Decode   = GRU(size_in=self.size_out, size=self.size)
        self.H0       = Zeros(size=self.size)
        self.params = sum([ l.params for l in [self.Encode, self.Decode, self.H0] ], [])


    def __call__(self, inp, out_prev):
        return self.Decode(last(self.Encode(self.H0(), inp, repeat_h0=1)), out_prev)
