import numpy as np

class perceptron:

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
            The output of the activation function. 1 if z > 0, -1 otherwise.
        """
        return 1 if z > 0 else -1
    
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
                pred.append(1 if self.act_fun(i) == 1 else 0)
        except:
            pred = self.act_fun(z)
            return 1 if pred == 1 else 0

        return pred
    
    def loss_function(self, prediction, target):
        """
        Calculate the loss between the predicted and target values.

        Parameters
        ----------
        prediction : float
            The predicted value.
        target : float
            The target value.

        Returns
        -------
        float
            The loss between the predicted and target values.
        """
        return target - prediction
    
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
        print(pred)
        loss = self.loss_function(pred, target)

        self.w[1:] += self.lr * loss * inputs
        self.w[0] += self.lr * loss * self.wb



    
    def fit(self, x, y, epochs):
        """
        Fitting the model to the data

        Parameters
        ----------
        x : array-like
            The input data.
        y : array-like
            The target values.
        epochs : int
            The number of epochs to train for.

        Returns
        -------
        None
        """
        
        for i in range(epochs):
            for inputs,target in zip(x,y):
                print(inputs,target)
                self.train(inputs, target)

    def get_weights(self):
        """
        Get the weights of the model.

        Returns
        -------
        w1 : float
            The weight for the first input.
        w2 : float
            The weight for the second input.
        w0 : float
            The bias weight.
        """

        return self.w[1], self.w[2], self.wb * self.w[0]