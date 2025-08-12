import numpy as np
import random as r

#in this file we regroup basic NN structures coded from scratch

class Linear:
    # this class is copied from my Transformer backward project
    def __init__(self, in_dim, out_dim, scale=1.0):
        self.W = self.xavier_init(in_dim, out_dim)
        self.b = np.zeros((out_dim,), dtype=np.float32)
        # gradients
        self.dW = np.zeros_like(self.W) #same dim as W
        self.db = np.zeros_like(self.b) #same dim as b
        self.cache = None #for backward we cache some data during forward
    
    def xavier_init(self, in_dim, out_dim):
        #initilize with xavier like method in order to respect numerical stability
        std = np.sqrt(2.0 / (in_dim + out_dim))
        return (np.random.randn(in_dim, out_dim).astype(np.float32) * std)

    def forward(self, x):
        self.x_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])  
        out_flat = x_flat @ self.W + self.b  
        out = out_flat.reshape(*x.shape[:-1], self.W.shape[1])
        # cache x_flat for backward
        self.cache = x_flat
        return(out)

    def backward(self, dout):
        # we reflatten the output
        dout_flat = dout.reshape(-1, dout.shape[-1]) 
        x_flat = self.cache  # (N, in_dim) #we use the flatten x that we cached in forward 
        # grads
        self.dW[:] = x_flat.T @ dout_flat 
        self.db[:] = dout_flat.sum(axis=0) 
        dx_flat = dout_flat @ self.W.T 
        dx = dx_flat.reshape(*dout.shape[:-1], self.W.shape[0])
        return(dx)

    def get_params(self):
        #get
        return([(self.W, self.dW, 'W'), (self.b, self.db, 'b')])
    
class Dropout:
    # we take the output of a layer and randomly put the value to 0 with respect to probability p
    def __init__(self, p, training=True):
        self.p = p
        self.cache = None #fro backward
        self.training = training

    def forward(self, x):
        if not self.training: #if we are not in training , just go through
            self.cache = np.ones_like(x)
            return(x)
        #entry: (b, seq, in_dim) or (N, out_dim)
        mask = np.random.binomial(1, 1 - self.p, size=x.shape).astype(x.dtype)
        #we create a mask that randomly 0es some elements in tensor x 
        self.cache = mask / (1 - self.p)
        return(x*self.cache) #hadamard
    
    def backward(self, dout):
        #we just remask the same elements
        #we dont want the droped out elements to influence the gradient of upward 
        return(dout*self.cache)

class ReLU:    
    def __init__(self):
        self.cache = None #cache for backprop
    @staticmethod
    def ReLU(x):
        #https://stackoverflow.com/questions/32109319/how-to-implement-the-relu-function-in-numpy
        return (x*(x > 0)) #fast, stable with numpy arrays
    
    def forward(self, x):
        self.cache = (x > 0).astype(x.dtype) #mask of the elements flatened 
        return(self.ReLU(x))
    
    def backward(self, dout):
        #we mask out the gradients
        return(dout*self.cache)
    
class Sigmoid:
    def __init__(self):
        self.cache = None #for backward
    @staticmethod
    def sigmoid(x):
        # we use this sigmoid for numerical stability
        # a previous simplier version was overflowing
        out = np.empty_like(x, dtype=np.float64)   # use float64 for numerical stability
        pos = x >= 0
        neg = ~pos
        # compute only where needed
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        exp_x = np.exp(x[neg])
        out[neg] = exp_x / (1.0 + exp_x)
        return out.astype(x.dtype)
    
    def forward(self, x):
        out = self.sigmoid(x)
        self.cache = out
        return(out)
    
    def backward(self, dout):
        # dy/dx = y(1-y)
        return(dout*(self.cache*(1-self.cache)))
    
class Tanh:
    def __init__(self):
        self.cache = None #for backward
    
    def forward(self, x):
        out = np.tanh(x)
        self.cache = out
        return(out)
    
    def backward(self, dout):
        # derivative : 1-tanh(x)**2
        return(dout*(1-self.cache**2))
    
class model:
    #a class that combine all the layers to make it be one model
    def __init__(self, layers):
        #layers is a list of all the layers in order
        self.layers = layers

    def forward(self, x):
        out = x.copy()
        for layer in self.layers:
            out = layer.forward(out)
        return(out)
    
    def backward(self, dout):
        for layer in self.layers[::-1]: #we go backward
            dout = layer.backward(dout)
        return(dout) #we dont really need to return, but gives info for debug