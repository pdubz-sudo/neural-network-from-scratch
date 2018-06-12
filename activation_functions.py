def sigmoid(Z):
    """
    Compute the sigmoid of Z

    Arguments:
    Z -- A scalar or numpy array of any size.

    Return:
    A -- sigmoid of Z
    backprop_store -- returns Z for backpropagation
    """

    A = 1/(1 + np.exp(-Z))
    backprop_store = Z

    return A, backprop_store

def leaky_relu(Z):
    """
    Compute leaky_ReLU of Z

    arguments:
    Z -- A scalar of numpy array of any size

    return:
    A -- post-activation of leaky ReLU of Z, same shape as Z
    backprop_store -- returns Z for backpropagation
    """

    A = np.maximum(0.01 * Z, Z)
    assert(A.shape == Z.shape)

    backprop_store = Z

    return A, backprop_store

def relu(Z):
    """
    Compute regular ReLU of Z

    arguments:
    Z -- A scalar of numpy array of any size

    return:
    A -- post-activation of ReLU of Z, same shape as Z
    backprop_store -- returns Z for backpropagation
    """

    A = np.maximum(0, Z)
    assert(A.shape == Z.shape)

    backprop_store = Z

    return A, backprop_store


def tanh(Z):
    """
    Compute tanh of Z

    arguments:
    Z -- A scalar of numpy array of any size

    return:
    A -- post-activation of tanh of Z, same shape as Z
    backprop_store -- returns Z for backpropagation
    """

    A = (np.exp(Z)-np.exp(-Z))  /  (np.exp(Z)+np.exp(-Z))
    assert(A.shape == Z.shape)

    backprop_store = Z

    return A, backprop_store


def softmax(Z):
    """
    Compute softmax of Z

    arguments:
    Z -- A scalar of numpy array of any size

    return:
    A -- post-activation of softmax of Z, same shape as Z
    backprop_store -- returns Z for backpropagation
    """

    e_Z = np.exp(Z - np.max(Z))
    A = e_Z / e_Z.sum()
    assert(A.shape == Z.shape)

    backprop_store = Z

    return A, backprop_store