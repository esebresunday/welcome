#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np


# In[22]:


class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        
    def forward_propagation(self, x_input):
        pass
    
    def backward_propagation(self, output_gradient, learning_rate):
        pass
    

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
    
    def forward_propagation(self, x_input):
        self.input = x_input 
        return np.dot(self.weights, self.input) + self.bias
    
    def backward_propagation(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        self.weights = self.weights - weights_gradient * learning_rate
        self.bias = self.bias - output_gradient * learning_rate

        return np.dot(self.weights.T, output_gradient)
    

class Activation(Layer):
    def __init__(self, function, function_derivative):
        self.function = function
        self.function_derivative = function_derivative
        
    def forward_propagation(self, x_input):
        self.input = x_input
        return self.function(self.input)
    
    def backward_propagation(self,output_gradient, learning_rate):
        return np.multiply(output_gradient, self.function_derivative(self.input))
    
def mean_squared_error(y_true, y_predicted):
    return np.mean((y_predicted - y_true)**2)

def mean_squared_error_derivative(y_true, y_predicted):
    n = np.size(y_true)
    return (2 / n) * (y_true - y_predicted)

class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_derivative = lambda x: 1 - np.tanh(x)**2
        super().__init__(tanh,tanh_derivative)
        
class Identity(Activation):
    def __init__(self):
        identity = lambda x: x
        identity_derivative = lambda x: 1
        super().__init__(identity,identity_derivative)
        
class Sigmoid(Activation):
    def __init__(self):
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        sigmoid_derivative = lambda x: np.exp(-x) / (1 + np.exp(-x))**2
        super().__init__(sigmoid,sigmoid_derivative)


# In[19]:


#to test the code, I define a network list containing the layers and I will loop through it
network = [
    Dense(1,5),
    Tanh(),
    Dense(5,1),
    Tanh()
]


# In[20]:


Y = np.array([
    [0.11],
    [0.29],
    [0.58],
    [1],
])
Y = np.reshape(Y, (4,1,1))

X = np.array([
    [0.25],
    [0.5],
    [0.75],
    [1],
])
X = np.reshape(X, (4,1,1))


# In[21]:


#the output of the code will print out the error and the epoch.
epochs = 100
learning_rate = 0.0001
for e in range(epochs):

    for x, y in zip(X,Y):
        output = x
        for layer in network:
            output = layer.forward_propagation(output) #this output contains an array of predicted y

        error = mean_squared_error(y, output)

        grad = mean_squared_error_derivative(y, output)
    
        for i in range(-1,-5,-1):
            if i != -4:
                grad = network[i].backward_propagation(grad, learning_rate)
            else:
                grad = network[i].backward_propagation(np.reshape(grad,(-1,1)), learning_rate)
                
            
        error /= len(X) 
    print("%d/%d, error = %f" %(e, epochs, error))


# In[ ]:





# In[ ]:




