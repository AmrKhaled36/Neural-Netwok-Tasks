import numpy as np

class Edge:
    lr = None
    def __init__(self,prev_neuron,nxt_neuron):
        
        self.weight = np.random.rand()
        self.prev_neuron = prev_neuron
        self.nxt_neuron = nxt_neuron

    def update_weight(self):
        if self.prev_neuron == None:
            self.weight += Edge.lr * self.nxt_neuron.error_signal
        else:
            self.weight += Edge.lr * self.nxt_neuron.error_signal * self.prev_neuron.output
    
    def calc_error_signal(self):
        return self.nxt_neuron.error_signal * self.weight


    