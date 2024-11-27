from edge import Edge
import numpy as np
from enum import Enum

class ActivationFunction(Enum):
    sigmoid = 0
    tanh = 1

class Neuron:

    
    def __init__(self,prev_layer, bias=0, activation_function= ActivationFunction.sigmoid):
        
        self.output = None
        self.target = None
        self.out_edges = []
        self.in_edges = []
        self.error_signal = None
        self.activation_function = activation_function

        if prev_layer != None:
            for i in range(len(prev_layer)):

                self.in_edges.append(Edge(prev_layer[i], nxt_neuron=self))
                prev_layer[i].out_edges.append(self.in_edges[-1])

            if bias:
                self.in_edges.append(Edge(None, nxt_neuron=self))
    
    def calc_net(self):

        net = 0
        for edge in self.in_edges:
            if edge.prev_neuron == None:
                net += edge.weight
                continue
            net += edge.prev_neuron.output * edge.weight

        return net

    def calc_output(self):

        net = self.calc_net()
        if self.activation_function == ActivationFunction.sigmoid:
            self.output = 1 / (1 + np.exp(-net))
        elif self.activation_function == ActivationFunction.tanh:
            self.output = np.tanh(net)
    
    def Act_derivative(self):

        if self.activation_function == ActivationFunction.sigmoid:
            return self.output * (1 - self.output)
        elif self.activation_function == ActivationFunction.tanh:
            return 1 - self.output ** 2
    
    def calc_error_signal(self):

        if len(self.out_edges) == 0:
            self.error_signal = (self.target - self.output) * self.Act_derivative()
        else:
            self.error_signal = 0
            for edge in self.out_edges:
                self.error_signal += edge.calc_error_signal()
            self.error_signal *= self.Act_derivative()

    def update_weight(self):
        for edge in self.in_edges:
            edge.update_weight()

        

        
        