class Network:
    def __init__(self):
        """
        Initialize a `layers` attribute to hold all the layers in the gradient tape.
        """
        self.layers = []
        self.input_layer = None
        self.output_layer = None

    def add(self, layer):
        """
        Adds a new layer to the network.

        Sublayers can *only* be added after their inputs have been added.
        (In other words, the DAG of the graph must be flattened and added in order from input to output)
        :param layer: The sublayer to be added
        """
        self.layers.append(layer)
        
    def set_input(self,input):
        """
        :param input: The sublayer that represents the signal input (e.g., the image to be classified)
        """
        self.input_layer = input

    def set_output(self,output):
        """
        :param output: SubLayer that produces the useful output (e.g., clasification decisions) as its output.
        """
        self.output_layer = output

    def forward(self,input):
        """
        Compute the output of the network in the forward direction, working through the gradient
        tape forward

        :param input: A torch tensor that will serve as the input for this forward pass
        :return: A torch tensor with useful output (e.g., the softmax decisions)
        """
        if not self.output_layer:
            raise Exception('Network must have output layer')
        
        self.input_layer.set(input)
        
        for layer in self.layers:
            layer.forward()
            
        try:
            return self.output_layer.classifications
        except:
            return self.output_layer.output

    def backward(self):
        """
        Compute the gradient of the output of all layers through backpropagation backward through the 
        gradient tape.

        """
        for layer in reversed(self.layers):
            layer.backward()

    def step(self, alpha):
        """
        Perform one step of the stochastic gradient descent algorithm
        based on the gradients that were previously computed by backward, updating all learnable parameters 

        """
        for layer in reversed(self.layers):
            layer.step(alpha)
            layer.clear_grad()
