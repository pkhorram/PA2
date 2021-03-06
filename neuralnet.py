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
import matplotlib.pyplot as plt



def load_config(path):
    """
    Load the configuration from config.yaml.
    """
    return yaml.load(open('config.yaml', 'r'), Loader=yaml.SafeLoader)


def normalize_data(data):
    data =  data.astype(np.float32)
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
    ans = np.zeros(x.shape)
    for i in range(x.shape[1]):
        temp = x[:,i]
        avg = np.average(temp)
        x1 = np.zeros(temp.shape) + temp - avg
        ans[:, i] = np.exp(x1) / np.sum(np.exp(x1))
        
    return ans
                
    
def accuracy(outputs, targets):
    
    out_ind = np.argmax(outputs, axis = 0)
    tar_ind = np.argmax(targets, axis = 0)
        
    acc = sum(out_ind == tar_ind)/outputs.shape[1]
    return acc*100



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

    def backward(self, delta):
        """
        Compute the backward pass.
        """
        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid(self.x)

        elif self.activation_type == "tanh":
            grad = self.grad_tanh(self.x)

        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU(self.x)

        return np.multiply(grad, delta)

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
        self.w = np.random.randn(out_units, in_units) #* np.sqrt(1 / in_units)    # Declare the Weight matrix
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
        self.d_b = np.sum(delta,axis=1, keepdims=True) + (penalty) * self.b 

        self.d_x = np.dot(self.w.T, delta)
        self.v_w = (momentum_gamma) * self.v_w + (1 - momentum_gamma) * self.d_w
        self.v_b = (momentum_gamma) * self.v_b + (1 - momentum_gamma) * self.d_b
        #self.w = self.w - (learning_rate) * self.d_w
        #self.b = self.b - (learning_rate) * self.d_b
        
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
        self.x = x
        self.targets = targets
        op = x
        for layer in self.layers:
            # print('init x shape: ', x.shape)
            op = layer.forward(op)
            # if isinstance(layer, Layer):
                # print('layer: ', x.shape, layer.w.shape, layer.b.shape)
            # elif isinstance(layer, Activation):
                # print('Activation: ', x.shape, layer.activation_type)

        # print('inforward: ', x.shape)
        self.y = softmax(op)
        #self.y = x
        # print('iny: ', np.sum(self.y), self.y.shape)

        if targets is not None:
            return self.y, self.loss(self.y, targets)
        return self.y

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
        error = self.loss(self.y, self.targets)
        #delta = self.y - self.targets
        #delta1 = - self.targets / self.y #check broadcast
        delta = (- self.targets + self.y)
        for i in range(len(self.layers)-1, -1, -1):
            if isinstance(self.layers[i], Activation):
                delta = self.layers[i].backward(delta)
            else:
                delta = self.layers[i].backwards(delta, self.momentum, self.momentum_gamma, self.penalty, self.learning_rate)
            
            


def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    Train your model here.
    Implement batch SGD to train the model.
    Implement Early Stopping.
    Use config to set parameters for training like learning rate, momentum, etc.
    """
    B = 100
    N = x_train.shape[0]
    num_epoch = 55

 
    # This dictionary strores the best parameters (Weights and Biases)
    models = {'L1_w':[],
             'L1_b':[],
             'L2_w':[],
             'L2_b':[]
             }

    val_l, train_l ,ind, train_acc, val_acc = [], [], [], [], []
    best_val_loss = 10
    for epoch in range(num_epoch):

        indeces = np.random.permutation(np.arange(N))

        for i in range(int(N/B)):

            xtrain = x_train[indeces[B*i:B*(i+1)],:]
            ytrain = y_train[indeces[B*i:B*(i+1)],:]
            train_forward, train_loss = model.forward(xtrain.T, ytrain.T)
            train_accuracy = accuracy(train_forward, ytrain.T)
            model.backward()


        val_forward, val_loss = model.forward(x_valid.T, y_valid.T)
        val_accuracy = accuracy(val_forward, y_valid.T)
        print('At Epoch', epoch, 'train_loss is: ', train_loss, 'val_loss is: ', val_loss)


        val_l.append(val_loss)
        train_l.append(train_loss)
        ind.append(epoch)
        train_acc.append(train_accuracy)
        val_acc.append(val_accuracy)

        # Storing the best loss and parameters 
        if val_l[epoch] < best_val_loss:
            models['L1_w'] = model.layers[0].w
            models['L1_b'] = model.layers[0].b
            models['L2_w'] = model.layers[2].w
            models['L2_b'] = model.layers[2].b
            

            best_val_loss = val_l[epoch]
            best_epoch = epoch

    fig1 = plt.figure()        
    plt.plot(ind, train_l,label = 'Train loss')
    plt.plot(ind, val_l,label = 'Validation loss')
    plt.title('Plot of Loss vs. Epoch for Batch Gradient Descent')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    
    fig2 = plt.figure()        
    plt.plot(ind, train_acc, label = 'Train Accuracy')
    plt.plot(ind, val_acc, label = 'Validation Accuracy')
    plt.title('Plot of Accuracy vs. Epoch for Batch Gradient Descent')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
     

    return best_val_loss, best_epoch, models
    
        


def test(models, X_test, y_test):
    """
    Calculate and return the accuracy on the test set.
    """
    keylist = list(models)
    X_test = np.dot(X_test, models[keylist[0]].T)
    X_test = X_test + models[keylist[1]].T
    act_layer = Activation(config['activation'])
    X_test = act_layer(X_test)
    X_test = np.dot(X_test, models[keylist[2]].T)
    X_test = X_test + models[keylist[3]].T
    
        
            
    
    X_test = softmax(X_test.T)
    print(X_test.shape)
    test_accuracy = accuracy(X_test, y_test)

    return test_accuracy


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
