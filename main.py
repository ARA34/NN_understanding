import numpy as np
inputs = [1,2,3, 2.5] # These are not neurons each, these are inputs to the neuron


weights = [0.2,0.8,-0.5,1.0]
bias = 2 

output = np.dot(weights, inputs) + bias
print(output)





