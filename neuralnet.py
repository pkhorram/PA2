################################################################################
# CSE 253: Programming Assignment 2
# Code snippet by Manjot Bilkhu
# Winter 2020
################################################################################
# We've provided you with the dataset in PA2.zip
################################################################################
# To install PyYaml, refer to the instructions for your system:
# https://pyyaml.org/wiki/PyYAMLDocumentation
################################################################################
# If you don't have NumPy installed, please use the instructions here:
# https://scipy.org/install.html
################################################################################

import os, gzip
import yaml
import numpy as np



def load_config(path):
    """
    Load the configuration from config.yaml.
    """
    return yaml.load(open('config.yaml', 'r'), Loader=yaml.SafeLoader)


def normalize_data(data):
    
    for i in range(data.shape[0]):
        maxim = np.max(data[i,:])
        minim = np.min(data[i,:])
        data[i,:] = (1/(maxim-minim))*data[i,:]
    return data
    


def one_hot_encoding(labels, num_classes=10):
    
    onehot = np.zeros((len(labels), num_classes))
    for ind, val in enumerate(labels):
        onehot[ind][val] = 1
    
    return onehot
    


def load_data(path, mode='train'):
    """
    Load Fashion MNIST data.
    Use mode='train' for train and mode='t10k' for test.
    """

    labels_path = os.path.join(path, f'{mode}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{mode}-images-idx3-ubyte.gz')

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    
    return images, labels


def softmax(x):
    """
    Implement the softmax function here.
    Remember to take care of the overflow condition.
    """
    avg = np.average(x)
    x1 = np.zeros(x.shape) - avg
    return np.exp(x1) / np.sum(np.exp(x1))


def softmaxp(predicted):
    
    n = predicted.shape[1]
    matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i,j] = predicted[i]*(1 - predicted[j])
            else:
                matrix[i,j] = -predicted[i]*predicted[j]
    return matrix
                
    




class Activation():
    """
    The class implements different types of activation functions for
    your neural network layers.

    Example (for sigmoid):
        >>> sigmoid_layer = Activation("sigmoid")
        >>> z = sigmoid_layer(a)
        >>> gradient = sigmoid_layer.backward(delta=1.0)
    """

    def __init__(self, activation_type = "sigmoid"):
        """
        Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU"]:
            raise NotImplementedError(f"{activation_type} is not implemented.")

        # Type of non-linear activation.
        self.activation_type = activation_type
        # Placeholder for input. This will be used for computing gradients.
        self.x = None

    def __call__(self, a):
        """
        This method allows your instances to be callable.
        """
        return self.forward(a)

    def forward(self, a):
        """
        Compute the forward pass.
        """
        self.x = a
        if self.activation_type == "sigmoid":
            return self.sigmoid(a)

        elif self.activation_type == "tanh":
            return self.tanh(a)

        elif self.activation_type == "ReLU":
            return self.ReLU(a)

    def backward(self, delta, momuntum, penalty):
        """
        Compute the backward pass.
        """
        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid(self.x)

        elif self.activation_type == "tanh":
            grad = self.grad_tanh(self.x)

        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU(self.x)

        return grad * delta

    def sigmoid(self, x):
        sigmoid = 1/(1+ np.exp(-x))
        return sigmoid

    def tanh(self, x):
        tanh_func = (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
        return tanh_func

    def ReLU(self, x):
        return x * (x > 0)

    
    def grad_sigmoid(self, x):
        return self.sigmoid(x)*(1 - self.sigmoid(x))

    def grad_tanh(self, x):
        return 1 - self.tanh(x)**2 

    def grad_ReLU(self, x):
        return 1 * ( x > 0)


class Layer():
    """
    This class implements Fully Connected layers for your neural network.

    Example:
        >>> fully_connected_layer = Layer(784, 100)
        >>> output = fully_connected_layer(input)
        >>> gradient = fully_connected_layer.backward(delta=1.0)
    """

    def __init__(self, in_units, out_units):
        """
        Define the architecture and create placeholder.
        """
        np.random.seed(42)
        self.w = np.random.randn(out_units, in_units) * np.sqrt(1 / in_units)    # Declare the Weight matrix
        self.b = np.zeros((out_units, 1))   # Create a placeholder for Bias
        self.x = None    # Save the input to forward in this
        self.a = None    # Save the output of forward pass in this (without activation)
        
        self.v_w = np.zeros((out_units, in_units))
        self.v_b = np.zeros((out_units, 1))

        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in thisc
        self.d_b = None  # Save the gradient w.r.t b in this
        

    def __call__(self, x):
        """
        Make layer callable.
        """
        return self.forward(x)

    def forward(self, x):
        """
        Compute the forward pass through the layer here.
        Do not apply activation here.
        Return self.a
        """
        self.x = x
        self.a = np.dot(self.w, self.x) + self.b
        return self.a
        #raise NotImplementedError("Layer forward pass not implemented.")

    def backwards(self, delta, momentum, momentum_gamma, penalty, learning_rate):
        """
        Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        Return self.dx
        """

        
        self.d_w = np.dot(delta, self.x.T) + (penalty) * self.w
        self.d_b = delta + (penalty) * self.b

        self.d_x = np.dot(self.w.T, delta)
        self.v_w = (momentum_gamma) * self.v_w + (1 - momentum_gamma) * self.d_w
        self.v_b = (momentum_gamma) * self.v_b + (1 - momentum_gamma) * self.d_b
        
        if momentum:
            self.w = self.w - (learning_rate) * self.v_w 
            self.b = self.b - (learning_rate) * self.v_b
        else:
            self.w = self.w - (learning_rate) * self.d_w
            self.b = self.b - (learning_rate) * self.d_b
        
        return self.d_x
        #raise NotImplementedError("Backprop for Layer not implemented.")


class Neuralnetwork():
    """
    Create a Neural Network specified by the input configuration.

    Example:
        >>> net = NeuralNetwork(config)
        >>> output = net(input)
        >>> net.backward()
    """

    def __init__(self, config):
        """
        Create the Neural Network using config.
        """
        self.layers = []     # Store all layers in this list.
        self.x = None        # Save the input to forward in this
        self.y = None        # Save the output vector of model in this
        self.targets = None  # Save the targets in forward in this variable

        # Add layers specified by layer_specs.
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i+1]))
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))
         
        self.momentum = config['momentum']
        self.penalty = config['L2_penalty']
        self.momentum_gamma = config['momentum_gamma']
        self.learning_rate = config['learning_rate']

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)

    def forward(self, x, targets):
        """
        Compute forward pass through all the layers in the network and return it.
        If targets are provided, return loss as well.
        """
        self.targets = targets
        
        for layer in self.layers:
            x = layer(x)

        x = softmax(x)
        self.y = x

        if targets is not None:
            return x, self.loss(x, targets)
        return x

    def loss(self, outputs, targets):
        '''
        compute the categorical cross-entropy loss and return it.
        assumes softmax probability distribution stored in logits 
        '''
        error = 0
        for i in range(outputs.shape[0]):
            error = error + np.dot(np.log(outputs[i,:]),targets[i,:])
        error = -error/(outputs.shape[0]*outputs.shape[1])
        return error

    def backward(self):
        '''
        Implement backpropagation here.
        Call backward methods of individual layer's.
        '''

        # delta = np.zeros((config['layer_specs'][-2],))
        error = loss(self.y, self.targets)
        #delta = self.y - self.targets
        delta1 = - self.targets / self.y #check broadcast
        delta = softmaxp(self.y, delta1)
        for i in range(len(self.layers)-1, -1, -1):
            if isinstance(model.layers[i], Activation):
                delta = self.layers[i].backward(delta)
            else:
                delta = self.layers[i].backwards(delta, self.momentum, self.momentum_gamma, self.penalty, self.learing_rate)
            
            


def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    Train your model here.
    Implement batch SGD to train the model.
    Implement Early Stopping.
    Use config to set parameters for training like learning rate, momentum, etc.
    """
    B = 100
    N = train_images.shape[0]
    num_epoch = 50

    for epoch in range(num_epoch):

        indeces = np.random.permutation(np.arange(N))

        for i in range(int(N/B)):

            xtrain = train_images[indeces[B*i:B*(i+1)],:]
            ytrain = train_labels[indeces[B*i:B*(i+1)],:]
            forwarded, loss = model.forward(xtrain.T, ytrain.T)
            model.backward()

        


def test(model, X_test, y_test):
    """
    Calculate and return the accuracy on the test set.
    """

    raise NotImplementedError("Test method not implemented")


if __name__ == "__main__":
    # Load the configuration.
    config = load_config("./")

    # Create the model
    model  = Neuralnetwork(config)

    # Load the data
    x_train, y_train = load_data(path="./", mode="train")
    x_test,  y_test  = load_data(path="./", mode="t10k")

    # Create splits for validation data here.
    # x_valid, y_valid = ...

    # train the model
    train(model, x_train, y_train, x_valid, y_valid, config)

    test_acc = test(model, x_test, y_test)
