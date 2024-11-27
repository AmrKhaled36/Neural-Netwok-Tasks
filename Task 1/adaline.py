import numpy as np

class adaline:

    def __init__(self,num_inputs, lr=0.01, bias=0):
        """
        Initialize the perceptron.

        Parameters
        ----------
        num_inputs : int
            The number of inputs to the perceptron.
        lr : float, optional
            The learning rate for the perceptron. Default is 0.01.
        bias : int, optional
            Whether to include a bias term in the perceptron. Default is 0.
        """

        self.w = np.random.rand(num_inputs + 1)
        if not bias:
            self.wb = 0
        else:
            self.wb = 1
        self.lr = lr

    def calc_weighted_sum(self, inputs):
        """
        Calculate the weighted sum of the inputs.

        Parameters
        ----------
        inputs : array-like
            The input data.

        Returns
        -------
        float
            The weighted sum of the inputs.
        """
        z = np.dot(inputs, self.w[1:]) + self.w[0] * self.wb
        return z
    
    def act_fun(self, z):
        """
        The activation function for the perceptron.
        Parameters
        ----------
        z : float
            The weighted sum of the inputs.

        Returns
        -------
        float
            The output of the linear activation function.
        """
        return z
    
    def prediction(self, inputs):
        """
        Make a prediction using the perceptron.

        Parameters
        ----------
        inputs : array-like
            The input data.

        Returns
        -------
        float
            The predicted value.
        """
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
        """
        Train the perceptron on a single data point.

        Parameters
        ----------
        inputs : array-like
            The input data.
        target : float
            The target value.

        Returns
        -------
        None
        """
        pred = self.prediction(inputs)
        loss = target - pred

        self.w[1:] += self.lr * loss * inputs
        self.w[0] += self.lr * loss * self.wb



    
    def fit(self, x, y, epochs, mse_threshold):
        """
        Fit the model to the data.

        Parameters
        ----------
        x : array-like
            The input data.
        y : array-like
            The target data.
        epochs : int
            The number of epochs to train for.
        mse_threshold : float
            The threshold for the mean squared error.

        Returns
        -------
        None
        """

        for i in range(epochs):
            MSE = 0
            for inputs,target in zip(x,y):
                self.train(inputs, target)

            for inputs,target in zip(x,y):
                MSE += (target - self.prediction(inputs))**2
            
            MSE = MSE/(2 * len(x))
            print(f"Epoch {i} loss:{MSE}")
            if MSE < mse_threshold:
                break

    def get_weights(self):
        """
        Returns the weights of the model.

        Returns
        -------
        float
            The weights of the model.
        """
        return self.w[1], self.w[2], self.wb * self.w[0]