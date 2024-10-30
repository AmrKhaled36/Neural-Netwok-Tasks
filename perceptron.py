import numpy as np
import GUI
class perceptron:

    def __init__(self,num_inputs, lr=0.01, bias=0):
        self.w = np.random.rand(num_inputs + 1)
        if not bias:
            self.wb = 0
        else:
            self.wb = 1
        self.lr = lr

    def calc_weighted_sum(self, inputs):
        z = np.dot(inputs, self.w[1:]) + self.w[0] * self.wb
        return z
    
    def act_fun(self, z, threshold=0):
        return 1 if z > threshold else 0
    
    def prediction(self, inputs, mse_threshold=0):
        z = self.calc_weighted_sum(inputs)
        return self.act_fun(z, mse_threshold)
    
    def loss_function(self, prediction, target):
        return prediction - target
    
    def train(self, inputs, target, mse_threshold=0):
        pred = self.prediction(inputs, mse_threshold)
        loss = self.loss_function(pred, target)

        self.w[1:] += self.lr * loss * inputs
        self.w[0] += self.lr * loss * self.wb

    
    def fit(self, x, y, epochs, mse_threshold=0):

        for i in range(epochs):
            for inputs,target in zip(x,y):
                print(inputs,target)
                self.train(inputs, target, mse_threshold=0)

    def get_weights(self):
        return self.w, self.wb