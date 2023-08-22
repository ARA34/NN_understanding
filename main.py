import numpy as np
import nnfs
from nnfs.datasets import spiral_data
#inputs = [1,2,3, 2.5] # These are not neurons each, these are inputs to the neuron

nnfs.init()


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0,inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs,axis=1,keepdims=True))
        probabilties = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilties



X,y = spiral_data(samples=100, classes=3)

# Inputs are two because data from x and data from y
dense1 = Layer_Dense(2,3) # inputs, outputs
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3,3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])




#biases = [2,3,0.5] 

# taking the dot product of weights and inputs yields the input*weights and then at end + bias
# matrix of vectors of weights * vectors of inputs

#output = np.dot(weights, inputs) + biases # numpy allows for (dot) function
#print(output)





