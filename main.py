import numpy as np
#inputs = [1,2,3, 2.5] # These are not neurons each, these are inputs to the neuron

inputs = [0,2,-1,3.3,-2.7,1.1,2.2,-100]

X = [
    [0.2,0.8,-0.5,1.0],
    [0.5,-0.91,0.26,-0.5],
    [-0.26,-0.27,0.17,0.87]
    ]


output = []
for i in inputs:
    if i > 0:
        output.append(i)
    elif i <= 0:
        output.append(0)

print(output)


#biases = [2,3,0.5] 

# taking the dot product of weights and inputs yields the input*weights and then at end + bias
# matrix of vectors of weights * vectors of inputs

#output = np.dot(weights, inputs) + biases # numpy allows for (dot) function
#print(output)





