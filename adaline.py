import numpy as np

class adaline:

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
    
    def act_fun(self, z):
        return z
    
    def prediction(self, inputs):
        z = self.calc_weighted_sum(inputs)
        
        try:
            pred = []
            for i in z:
                pred.append(self.act_fun(i))
        except:
            pred = self.act_fun(z)
            return pred

        return pred
    
    
    def train(self, inputs, target):
        pred = self.prediction(inputs)
        loss = target - pred

        self.w[1:] += self.lr * loss * inputs
        self.w[0] += self.lr * loss * self.wb

        #print(f"pred: {pred} loss: {loss}, w: {self.w}, wb: {self.wb}")



    
    def fit(self, x, y, epochs, mse_threshold):

        for i in range(epochs):
            MSE = 0
            for inputs,target in zip(x,y):
                self.train(inputs, target)
                MSE += (target - self.prediction(inputs))**2
            MSE = MSE/(2 * len(x))
            print(f"Epoch {i} loss:{MSE}")
            if MSE < mse_threshold:
                break

    def get_weights(self):
        return self.w[1], self.w[2], self.wb * self.w[0]