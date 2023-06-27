import torch
from numpy import newaxis as np_newaxis

# Layers in this file are arranged in roughly the order they
# would appear in a network.


class Layer:
    def __init__(self, shape):
        self.shape = shape
        self.output = torch.zeros(self.shape)
        self.grad = torch.zeros(self.shape)

    def accumulate_grad(self, grad):
        """
        Add arguments as needed for this method.
        This method should accumulate its grad attribute with the value provided.
        """
        assert self.grad.shape == grad.shape, "Gradient shapes must match"
        self.grad += grad

    def clear_grad(self):
        """
        Add arguments as needed for this method.
        This method should clear grad elements. It should set the grad to the right shape 
        filled with zeros.
        """
        self.grad = torch.zeros(self.shape)

    def step(self, alpha):
        """
        Add arguments as needed for this method.
        Most tensors do nothing during a step so we simply do nothing in the default case.
        """
        pass

class Input(Layer):
    def __init__(self, shape, train):
        """
        Input layers with single dimensions can be declared like any of the following:
            integer: input_size
            tuple: (input_size,)
            tuple: (input_size,1)
        """
        if isinstance(shape, int):
            shape = (shape, 1)
        elif len(shape) == 1:
            shape = (shape[0], 1)
        Layer.__init__(self, shape)
        self.train = train

    def set(self, output):
        """
        Accept any arguments specific to this method.
        Set the output of this input layer.
        :param output: The output to set, as a torch tensor. Raise an error if this output's size
                       would change.
        """
        if len(output.shape) > 1:
            assert output.shape == self.shape, "Shapes do not match"
            self.output = output
        else:
            assert output.shape[0] == self.shape[0], "Shapes do not match"
            self.output = output.unsqueeze(1)
        

    def randomize(self):
        """
        Accept any arguments specific to this method.
        Set the output of this input layer to random values sampled from the standard normal
        distribution (torch has a nice method to do this). Ensure that the output does not
        change size.
        """
        self.output = torch.randn(self.shape)

    def forward(self):
        """
        Accept any arguments specific to this method.
        Set this layer's output based on the outputs of the layers that feed into it.
        """
        pass

    def backward(self):
        """
        Accept any arguments specific to this method.
        This method does nothing as the Input layer should have already received its output
        gradient from the previous layer(s) before this method was called.
        """
        pass

    def step(self, alpha):
        """
        Add arguments as needed for this method.
        This method should have a precondition that the gradients have already been computed
        for a given batch.

        It should perform one step of stochastic gradient descent, updating the weights of
        this layer's output based on the gradients that were computed and a learning rate.
        """
        if self.train:
            self.output -= alpha * self.grad

class Linear(Layer):
    def __init__(self, W: Input, b: Input, x: Layer):
        """
        Accept any arguments specific to this child class.
        """
        assert W.shape[1] == x.shape[0]
        assert W.shape[0] == b.shape[0]
        Layer.__init__(self, (W.shape[0], x.shape[1]))
        self.W = W
        self.b = b
        self.x = x

    def forward(self):
        """
        Accept any arguments specific to this method.
        Set this layer's output based on the outputs of the layers that feed into it.
        """
        self.output = self.W.output@self.x.output + self.b.output

    def backward(self):
        """
        Accept any arguments specific to this method.
        This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """
        self.W.accumulate_grad(self.grad@self.x.output.T)
        self.b.accumulate_grad(self.grad)
        self.x.accumulate_grad(self.W.output.T@self.grad)


class ReLU(Layer):
    def __init__(self, u: Linear):
        """
        Accept any arguments specific to this child class.
        """
        Layer.__init__(self, u.shape)
        self.u = u

    def forward(self):
        """
        Accept any arguments specific to this method.
        Set this layer's output based on the outputs of the layers that feed into it.
        """
        self.output = self.u.output * (self.u.output > 0)

    def backward(self):
        """
        Accept any arguments specific to this method.
        This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """
        self.u.accumulate_grad(self.grad * ((self.u.output > 0) * torch.ones(self.u.shape)))


class MSELoss(Layer):
    """
    This is a good loss function for regression problems.
    It implements the MSE norm of the inputs.
    """
    def __init__(self, v: Layer, y: Input): 
        """
        Accept any arguments specific to this child class.
        """
        assert v.shape == y.shape, "Shapes do not match"
        Layer.__init__(self, 1)
        self.v = v
        self.y = y

    def forward(self):
        """
        Accept any arguments specific to this method.
        Set this layer's output based on the outputs of the layers that feed into it.
        """
        self.output = sum((self.v.output - self.y.output)**2) / len(self.v.output)

    def backward(self):
        """
        Accept any arguments specific to this method.
        This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """
        self.v.accumulate_grad(self.grad * (self.v.output - self.y.output))


class Regularization(Layer):
    def __init__(self, W: Input, lam: int):
        """
        Accept any arguments specific to this child class.
        """
        Layer.__init__(self, 1)
        self.W = W
        self.lam = lam

    def forward(self):
        """
        Accept any arguments specific to this method.
        Set this layer's output based on the outputs of the layers that feed into it.
        """
        self.output = self.lam * (self.W.output.norm()**2)

    def backward(self):
        """
        Accept any arguments specific to this method.
        This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """
        self.W.accumulate_grad(self.lam * self.W.output)


class Softmax(Layer):
    """
    This layer is an unusual layer.  It combines the Softmax activation and the cross-
    entropy loss into a single layer.

    The reason we do this is because of how the backpropagation equations are derived.
    It is actually rather challenging to separate the derivatives of the softmax from
    the derivatives of the cross-entropy loss.

    So this layer simply computes the derivatives for both the softmax and the cross-entropy
    at the same time.

    But at the same time, it has two outputs: The loss, used for backpropagation, and
    the classifications, used at runtime when training the network.

    Create a self.classifications property that contains the classification output,
    and use self.output for the loss output.

    See https://www.d2l.ai/chapter_linear-networks/softmax-regression.html#loss-function
    in our textbook.

    Another unusual thing about this layer is that it does NOT compute the gradients in y.
    We don't need these gradients for this lab, and usually care about them in real applications,
    but it is an inconsistency from the rest of the lab.
    """
    def __init__(self, v: Layer, y: Input):
        """
        Accept any arguments specific to this child class.
        """
        Layer.__init__(self, 1)
        self.v = v
        self.y = y
        self.classifications = torch.zeros(v.output.shape)

    def forward(self):
        """
        Accept any arguments specific to this method.
        Set this layer's output based on the outputs of the layers that feed into it.
        """
        v_exp = torch.exp(self.v.output - self.v.output.max(dim=0, keepdim=True).values)
        self.classifications = v_exp / (v_exp.sum(axis=0, keepdim=True))
        self.output = -(self.y.output * torch.log(self.classifications + 1e-9)).sum(dim=0)

    def backward(self):
        """
        Accept any arguments specific to this method.
        Set this layer's output based on the outputs of the layers that feed into it.
        This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """
        self.v.accumulate_grad(self.grad * (self.classifications - self.y.output))


class Sum(Layer):
    def __init__(self, *args):
        """
        Accept any arguments specific to this child class.
        """
        Layer.__init__(self, args[0].shape)
        self.layers = []
        for arg in args:
            assert isinstance(arg, Layer), "All inputs must be Layers"
            assert arg.shape == self.shape, "All input shapes must match"
            self.layers.append(arg)

    def forward(self):
        """
        Accept any arguments specific to this method.
        Set this layer's output based on the outputs of the layers that feed into it.
        """
        self.output = sum(layer.output for layer in self.layers)

    def backward(self):
        """
        Accept any arguments specific to this method.
        This network's grad attribute should already have been accumulated before calling
        this method.  This method should compute its own internal and input gradients
        and accumulate the input gradients into the previous layer.
        """
        for layer in self.layers:
            layer.accumulate_grad(torch.ones(self.shape))

