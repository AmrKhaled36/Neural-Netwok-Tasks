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
    
    def act_fun(self, z, threshold):
        return 1 if z > threshold else -1
    
    def prediction(self, inputs, mse_threshold):
        z = self.calc_weighted_sum(inputs)
        
        try:
            pred = []
            for i in z:
                pred.append(1 if self.act_fun(i, mse_threshold) == 1 else 0)
        except:
            pred = self.act_fun(z, mse_threshold)
            return 1 if pred == 1 else 0

        return pred
    
    def loss_function(self, prediction, target):
        return target - prediction
    
    def train(self, inputs, target, mse_threshold):
        pred = self.prediction(inputs, mse_threshold)
        loss = self.loss_function(pred, target)

        self.w[1:] += self.lr * loss * inputs
        self.w[0] += self.lr * loss * self.wb

        #print(f"pred: {pred} loss: {loss}, w: {self.w}, wb: {self.wb}")



    
    def fit(self, x, y, epochs, mse_threshold=0):

        for i in range(epochs):
            for inputs,target in zip(x,y):
                print(inputs,target)
                self.train(inputs, target, mse_threshold)

    def get_weights(self):
        return self.w[1], self.w[2], self.wb * self.w[0]