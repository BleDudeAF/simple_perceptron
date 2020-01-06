import numpy as np
repetitions = 50000

#we want to train the following truth table: [0,0,1] = 0, [1,1,1] = 1, [1,0,1] = 1,[0,1,1] = 0

#normalizing function phi, maps outputs to range 0 to 1. Using a sigmoid
def sigmoid(x):
    return 1/(1+np.exp(-x))

def nablasigmoid(sig_of_x):
    #return np.exp(-x)/np.square((np.exp(-x)+1)) #it holds that: sig(x)' = sig(x)(1-sig(x)), thus
    return sig_of_x*(1-sig_of_x) 


#input training data, l0 layer
training_inputs = np.array([[0,0,1], [1,1,1], [1,0,1],[0,1,1]])

#output training data for supervised learning, l3 layer (while testing)
training_outputs = np.array([[0,1,1,0]]).T

#lets start with a random weight distribution
np.random.seed(1)
weights_l1 = 2 * np.random.random((3,4)) -1
weights_l2 = np.random.random((4,1)) -1
print('random starting synaptic weights: ')
print(weights_l1)
print(weights_l2)

#train the neural network
for iter in range(10000):

    input_layer = training_inputs

    #calc the predictions and propagation through both hidden layers: phi(x) = sigmoid(sigma(xi*wi) i from 1 to 3), sigmoid as normalizing function
    l1 = sigmoid(np.dot(input_layer, weights_l1)) 
    l2 = sigmoid(np.dot(l1,weights_l2))

    l2_err = training_outputs - l2  #error
    l2_weighted_err = l2_err*nablasigmoid(l2)   #normalized error wrt layer 2
    l1_err = l2_weighted_err.dot(weights_l2.T)  #impact of 1st layer increases if onset regarded and weight of 2nd layer near 1 (if we want to achieve an "1" with the nodes looked at, we want the weight to be near 1)
    l1_weighted_err = l1_err*nablasigmoid(l1)

    weights_l2 += l1.T.dot(l2_weighted_err)
    weights_l1 += input_layer.T.dot(l1_weighted_err)

    #adjustment = np.dot(input_layer.T, errors*nablasigmoid(outputs)) #errors*nablasigmoid is an element wise vector multiplication (Hadamard)


print('outputs after training')
print(l2)

#test NN    

propagation1 = sigmoid(np.dot(np.array([[1,0,0]]), weights_l1))
propagation2 = sigmoid(np.dot(propagation1, weights_l2))
print(propagation2)
