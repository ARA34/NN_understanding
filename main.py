import numpy as np
import nnfs
from nnfs.datasets import spiral_data
#inputs = [1,2,3, 2.5] # These are not neurons each, these are inputs to the neuron

nnfs.init()

inputs = [0,2,-1,3.3,-2.7,1.1,2.2,-100]

#X = [
 #   [0.2,0.8,-0.5,1.0],
  #  [0.5,-0.91,0.26,-0.5],
   # [-0.26,-0.27,0.17,0.87]
    #]

X,y = spiral_data(100,3)



class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self, inputs):
        self.output = np.maximum(0,inputs)

class Activation_ReLU:
    def forward(self, input):
        self.output = np.maximum(0,inputs)

# Inputs are two because data from x and data from y
layer1 = Layer_Dense(2,5) # inputs, neurons
activation1 = Activation_ReLU()
layer1.forward(X)
#print(layer1.output)
activation1.forward(layer1.output)
print(activation1.output)




#biases = [2,3,0.5] 

# taking the dot product of weights and inputs yields the input*weights and then at end + bias
# matrix of vectors of weights * vectors of inputs

#output = np.dot(weights, inputs) + biases # numpy allows for (dot) function
#print(output)





