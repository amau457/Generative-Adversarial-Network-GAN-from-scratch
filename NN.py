import numpy as np
import random as r

#in this file we regroup basic NN structures coded from scratch

class Linear:
    # this class is copied from my Transformer backward project
    def __init__(self, in_dim, out_dim, scale=1.0):
        self.W = np.random.randn(in_dim, out_dim).astype(np.float32) * scale
        self.b = np.zeros((out_dim,), dtype=np.float32)
        # gradients
        self.dW = np.zeros_like(self.W) #same dim as W
        self.db = np.zeros_like(self.b) #same dim as b
        self.cache = None #for backward we cache some data during forward

    def forward(self, x):
        # x shape: (b, seq, in_dim)  OR (N, in_dim) N=batch*seq
        self.x_shape = x.shape
        # out = x @ self.W + self.b
        # it does the same thing as flattening and then doing the operation
        # but it is restrictive for dimensions
        # it will be simplier to do the backpropagation if we used the flattened (reduced to 2 dim) tensors
        x_flat = x.reshape(-1, x.shape[-1])  # (N, in_dim)
        out_flat = x_flat @ self.W + self.b  # (N, out_dim)
        out = out_flat.reshape(*x.shape[:-1], self.W.shape[1])
        # cache x_flat for backward
        self.cache = x_flat
        return(out)

    def backward(self, dout):
        # we reflatten the output
        dout_flat = dout.reshape(-1, dout.shape[-1])  # (N, out_dim)
        x_flat = self.cache  # (N, in_dim) #we use the flatten x that we cached in forward 
        # grads
        self.dW[:] = x_flat.T @ dout_flat # copy (in_dim, out_dim)
        self.db[:] = dout_flat.sum(axis=0) #copy (out_dim,)
        dx_flat = dout_flat @ self.W.T # (N, in_dim)
        dx = dx_flat.reshape(*dout.shape[:-1], self.W.shape[0])
        return(dx)

    def get_params(self):
        #get
        return([(self.W, self.dW, 'W'), (self.b, self.db, 'b')])
    
class dropout:
    # we take the output of a layer and randomly put the value to 0 with respect to probability p
    def __init__(self, p):
        self.p = p
        self.cache = None #fro backward

    def forward(self, x):
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
        #https://stackoverflow.com/questions/51976461/optimal-way-of-defining-a-numerically-stable-sigmoid-function-for-a-list-in-pyth
        return np.where(
                x >= 0, # condition
                1 / (1 + np.exp(-x)), # For positive values
                np.exp(x) / (1 + np.exp(x)) # For negative values
                )
    
    def forward(self, x):
        out = self.sigmoid(x)
        self.cache = out
        return(out)
    
    def backward(self, dout):
        # dy/dx = y(1-y)
        return(dout*(self.cache*(1-self.cache)))
    
class tanh:
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