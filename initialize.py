def initialize_parameters(initialize, dimension_of_layers):
    """
    Arguments:
    initialization -- activation used in this layer. 
        Stored as text string: "He", "Xavier", "Yoshua" "random"
    dimensions_of_layers -- array (list) of size in each layer

    Returns:
    parameters -- dictionary containing parameters "W1", "b1", "W2", "b2",...
                W[layer] -- shape (dimension_of_layers[layer], (dimension_of_layers[layer-1])
                b[layer] -- bias vector shape (dimension_of_layers[layer], 1) 
    """
    
    # np.random.seed(1)  # Use when you need to test that the different initializations are giving different numbers
    parameters = {}
    num_layers = len(dimension_of_layers)

    for layer in range(1, num_layers):  # this will loop through first hidden layer to final output layer

        if initialize == "He":
            parameters["W" + str(layer)] = np.random.randn(dimension_of_layers[layer], 
                dimension_of_layers[layer - 1]) * np.sqrt(2. / dimension_of_layers[layer - 1])
            parameters["b" + str(layer)] = np.zeros( (dimension_of_layers[layer], 1) )

        elif initialize == "Yoshua":
            parameters["W" + str(layer)] = np.random.randn(dimension_of_layers[layer], 
                dimension_of_layers[layer - 1]) * np.sqrt(2. / (dimension_of_layers[layer - 1] + dimension_of_layers[layer]))
            parameters["b" + str(layer)] = np.zeros( (dimension_of_layers[layer], 1) )

        elif initialize == "Xavier":
            parameters["W" + str(layer)] = np.random.randn(dimension_of_layers[layer], 
                dimension_of_layers[layer - 1]) * np.sqrt(1. / (dimension_of_layers[layer - 1]))
            parameters["b" + str(layer)] = np.zeros( (dimension_of_layers[layer], 1) )

        elif initialize == "random":
            parameters["W" + str(layer)] = np.random.randn(dimension_of_layers[layer], dimension_of_layers[layer - 1]) * 0.01
            parameters["b" + str(layer)] = np.zeros( (dimension_of_layers[layer], 1) )

        else:
            print("ERROR: YOU MUST CHOOSE AN INITIALIZATION TYPE")

            assert parameters["weights" + str(layer)].shape == (dimension_of_layers[layer], dimension_of_layers[layer - 1])
            assert parameters["bias" + str(layer)].shape == (dimension_of_layers[layer], 1)

    return parameters