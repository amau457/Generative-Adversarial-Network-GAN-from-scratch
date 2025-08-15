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
    
class MinibatchDiscrimination:
    # minibatch discrimination layer
    # if we take an entry x (size: (N, D))
    # we do M = x @ T we T a tensor of size (N, M*K) with K nb of kernel and M sizes of kernels
    # we calculate the similarity L1 for every example i and kernel k
    # sum_j (exp(-sum_p(abs(M[i,k,p]-M[j,k,p])))) we note it c_i,k
    # we return [x, c] the concatenation (with respect of dim)
    # x (N,D), c(N, D+K)
    def __init__(self, in_dim, num_kernels=32, kernel_dim=4, init_scale=0.02):
        self.in_dim = in_dim
        self.K = int(num_kernels)
        self.M = int(kernel_dim)
        # T shape (D, K*M)
        self.T = (np.random.randn(in_dim, self.K * self.M).astype(np.float32) * init_scale)
        self.dT = np.zeros_like(self.T, dtype=np.float32)
        self.cache = None

    def forward(self, x):
        # x: (N, D)
        x_flat = x.reshape(-1, x.shape[-1]).astype(np.float32)
        N = x_flat.shape[0]
        M_flat = x_flat @ self.T
        M = M_flat.reshape(N, self.K, self.M)

        # pairwise L1 distances per kernel
        diff = M[:, None, :, :] - M[None, :, :, :]               
        absdiff = np.sum(np.abs(diff), axis=3)                  
        weight = np.exp(-absdiff) #we calculate the exp
        # set diagonal to 0 so that c_i doesn't include j==i, exclude self contribution
        if N > 1:
            for k in range(self.K):
                np.fill_diagonal(weight[:, :, k], 0.0)

        # c sum over j
        c = np.sum(weight, axis=1)
        out = np.concatenate([x_flat, c], axis=1) #(N, D + K)

        # cache for backward
        # store x_flat, M, weight, diff (we need sign(diff) in backward)
        sign = np.sign(diff)
        self.cache = (x_flat, M, weight, sign, N)
        return(out.reshape(*x.shape[:-1], out.shape[-1]))

    def backward(self, dout):
        x_flat, M, weight, sign, N = self.cache
        N = x_flat.shape[0]
        D = self.in_dim
        dout_flat = dout.reshape(N, -1)
        # split gradients
        g_x_direct = dout_flat[:, :D] # direct gradient on input x (identity passthrough)
        g_c = dout_flat[:, D:]  # gradient on the minibatch features c

        # compute gradient / M
        grad_M = np.zeros_like(M, dtype=np.float32) 
        # for each kernel k, compute contribution:
        # grad_M[i,k,:] += - sum_j weight[i,j,k] * sign[i,j,k,:] * g_c[i,k]
        # grad_M[i,k,:] +=   sum_j weight[j,i,k] * sign[i,j,k,:] * g_c[j,k]
        for k in range(self.K):
            w_k = weight[:, :, k] # (N,N)
            s_k = sign[:, :, k, :]  # (N,N,M)
            gck = g_c[:, k].astype(np.float32)   # (N,)

            # term A= - sum_j w_ij * sign_ij * g_c[i]
            # compute sum_j (w_ij * sign_ij) 
            sum_w_sign = np.sum(w_k[:, :, None] * s_k, axis=1)  # (N, M)
            termA = - (gck[:, None] * sum_w_sign)
            # term B: sum_j w_ji * sign_ij * g_c[j]
            # weight_ji = w_k.T ; we want sum_j (w_ji * g_c[j]) * sign_ij
            wji_gc = (w_k.T * gck[None, :])
            termB = np.sum(wji_gc[:, :, None] * s_k, axis=1)
            grad_M[:, k, :] = termA + termB

        # flatten grad_M to shape (N, K*M)
        grad_M_flat = grad_M.reshape(N, -1)   # (N, K*M)

        self.dT[:] = (x_flat.T @ grad_M_flat).astype(np.float32)
        dx_from_M = (grad_M_flat @ self.T.T).astype(np.float32)

        dx = g_x_direct + dx_from_M
        return(dx.reshape(*dout.shape[:-1], D))

    def get_params(self):
        return([(self.T, self.dT, 'T')])