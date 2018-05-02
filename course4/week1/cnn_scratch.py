import numpy as np
import math

def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image, 
    as illustrated in Figure 1.
    
    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions
    
    Returns:
    X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    """
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
    return X_pad

def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
    of the previous layer.
    
    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
    
    Returns:
    Z -- a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data
    """
    Z = np.sum(a_slice_prev * W + b)
    return Z

def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function
    
    Arguments:
    A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"
        
    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function, namely all this function's parameters
    """
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    f = W.shape[0]
    n_C = W.shape[3]
    A_prev_pad = zero_pad(A_prev, pad)
    m, n_H_prev_pad, n_W_prev_pad, _ = A_prev_pad.shape
    n_H = math.floor((n_H_prev_pad - f) / stride) + 1
    n_W = math.floor((n_W_prev_pad - f) / stride) + 1
    Z = np.zeros((m, n_H, n_W, n_C))
    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    A_slice_prev = np.squeeze(A_prev_pad[i, h*stride:h*stride+f, w*stride:w*stride+f, :])
                    W_channel = np.squeeze(W[:, :, :, c])
                    Z[i, h, w, c] = conv_single_step(A_slice_prev, W_channel, b[:, :, :, c])
    cache = (A_prev, W, b, hparameters)
    return Z, cache

def pool_forward(A_prev, hparameters, mode = "max"):
    """
    Implements the forward pass of the pooling layer
    
    Arguments:
    A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    hparameters -- python dictionary containing "f" and "stride"
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
    A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters 
    """
    m, n_H_prev, n_W_prev, n_C = A_prev.shape
    f = hparameters["f"]
    stride = hparameters["stride"]
    n_H = math.floor((n_H_prev - f) / stride) + 1
    n_W = math.floor((n_W_prev - f) / stride) + 1
    A = np.zeros((m, n_H, n_W, n_C))
    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    A_slice_prev = np.squeeze(A_prev[i, h*stride:h*stride+f, w*stride:w*stride+f, c])
                    if mode == "max":
                        A[i, h, w, c] = np.max(A_slice_prev)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(A_slice_prev)
    cache = (A_prev, hparameters)
    return A, cache

def conv_backward(dZ, cache):
    """
    Implement the backward propagation for a convolution function
    
    Arguments:
    dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward(), output of conv_forward()
    
    Returns:
    dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
               numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    dW -- gradient of the cost with respect to the weights of the conv layer (W)
          numpy array of shape (f, f, n_C_prev, n_C)
    db -- gradient of the cost with respect to the biases of the conv layer (b)
          numpy array of shape (1, 1, 1, n_C)
    """
    A_prev, W, b, hparameters = cache
    m, n_H, n_W, n_C = dZ.shape
    f = W.shape[0]
    pad = hparameters["pad"]
    stride = hparameters["stride"]
    _, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))
    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    dA_prev_pad[i, h*stride:h*stride+f, w*stride:w*stride+f, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += A_prev_pad[i, h*stride:h*stride+f, w*stride:w*stride+f, :] * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]
    dA_prev = dA_prev_pad[:, pad:-pad, pad:-pad, :]             
    return dA_prev, dW, db

def create_mask_from_window(x):
    """
    Creates a mask from an input matrix x, to identify the max entry of x.
    
    Arguments:
    x -- Array of shape (f, f)
    
    Returns:
    mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
    """
    return x == np.max(x)

def distribute_value(dz, shape):
    """
    Distributes the input value in the matrix of dimension shape
    
    Arguments:
    dz -- input scalar
    shape -- the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz
    
    Returns:
    a -- Array of size (n_H, n_W) for which we distributed the value of dz
    """
    n_H, n_W = shape
    average = dz / (n_H * n_W)
    return np.ones((n_H, n_W)) * average

def pool_backward(dA, cache, mode = "max"):
    """
    Implements the backward pass of the pooling layer
    
    Arguments:
    dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
    cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters 
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
    dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
    """
    A_prev, hparameters = cache
    f = hparameters["f"]
    stride = hparameters["stride"]
    dA_prev = np.zeros(A_prev.shape)
    m, n_H, n_W, n_C = dA.shape
    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    a_slice_prev = A_prev[i, h*stride:h*stride+f, w*stride:w*stride+f, c]
                    if mode == "max":
                        mask = create_mask_from_window(a_slice_prev)
                        dA_prev[i, h*stride:h*stride+f, w*stride:w*stride+f, c] += dA[i, h, w, c] * mask
                    elif mode == "average":
                        dA_prev[i, h*stride:h*stride+f, w*stride:w*stride+f, c] += distribute_value(dA[i, h, w, c], (f, f))                    
    return dA_prev
    
    
    
    
    
    