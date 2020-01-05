import numpy as np
repetitions = 10000

#we want to train the following truth table: [0,0,1] = 0, [1,1,1] = 1, [1,0,1] = 1,[0,1,1] = 0

#normalizing function phi, maps outputs to range 0 to 1. Using a sigmoid
def sigmoid(x):
    return 1/(1+np.exp(-x))

def nablasigmoid(x):
    return np.exp(-x)/np.square((np.exp(-x)+1))

#input training data
training_inputs = np.array([[0,0,1], [1,1,1], [1,0,1],[0,1,1]])

#output training data for supervised learning
training_outputs = np.array([[0,1,1,0]]).T

#lets start with a random weight distribution
np.random.seed(1)
synaptic_weights = 2 * np.random.random((3,1)) -1
print('random starting synaptic weights: ')
print(synaptic_weights)

#train the neural network
for iter in range(repetitions):

    input_layer = training_inputs #load inputs

    #calc the first predictions based on the random weights
    #phi(x) = sigmoid(sigma(xi*wi) i from 1 to 3)
    outputs = sigmoid(np.dot(input_layer, synaptic_weights)) 
    
    errors = training_outputs-outputs
    
    #input_layer is 4x3 matrix -> adjustment is 4x1 vector from matrix vector multiplication
    adjustment = np.dot(input_layer.T, errors*nablasigmoid(outputs)) #errors*nablasigmoid is an element wise vector multiplication (Hadamard)
    synaptic_weights += adjustment

print('outputs after training')
print(outputs)

#test the nn:
test_data = np.array([[1,0,1]])
print(sigmoid(np.dot(test_data, synaptic_weights)))