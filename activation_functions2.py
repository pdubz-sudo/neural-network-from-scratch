def activation_function(Z, activation_string):
    """
    Compute the activation function of Z

    Arguments:
    Z -- A scalar or numpy array of any size.
    activation_string -- the activation to be used in this layer, stored as a text string: 
    "sigmoid", "relu", "leaky relu", "tanh", "softmax"

    Return:
    A -- sigmoid of Z
    backprop_store_lin -- returns Z for backpropagation
    """


    if activation_string == "sigmoid":
        # Inputs: "Z". Outputs: "A, backprop_store_lin".
        A = 1/(1 + np.exp(-Z))
        backprop_store_lin = Z

    elif activation_string == "leaky relu":
        # Inputs: "Z". Outputs: "A, backprop_store_lin".
        A = np.maximum(0.01 * Z, Z)
        backprop_store_lin = Z

    elif activation_string == "relu":
        # Inputs: "Z". Outputs: "A, backprop_store_lin".
        A = np.maximum(0, Z)
        backprop_store_lin = Z

    elif activation_string == "tanh":
        # Inputs: "Z". Outputs: "A, backprop_store_lin".
        A = (np.exp(Z)-np.exp(-Z))  /  (np.exp(Z)+np.exp(-Z))
        backprop_store_lin = Z

    elif activation_string == "softmax":
        # Inputs: "Z". Outputs: "A, backprop_store_lin".
        e_Z = np.exp(Z - np.max(Z))
        A = e_Z / e_Z.sum()
        backprop_store_lin = Z

    assert(A.shape == Z.shape)

    return A, backprop_store_lin