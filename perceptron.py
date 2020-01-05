import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

training_inputs = np.array([[0,0,1], [1,1,1], [1,0,1],[0,1,1]])

training_outputs = np.array([[0,1,1,0]]).T

print(training_inputs)
print(training_outputs)