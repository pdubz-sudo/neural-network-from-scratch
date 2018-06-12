def linear_forward(A_prev, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)

    Returns:
    A_post -- the input of the activation function, also called pre-activation parameter 
    backprop_store -- a python dictionary containing "W", "b", and "A_prev" ; stored for computing the backward pass efficiently
    """

    Z = np.dot(W, A_prev) + b
    
    assert(Z.shape == (W.shape[0], A_prev.shape[1]))
    backprop_store = (W, b, A_prev)
    
    return Z, backprop_store