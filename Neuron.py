import math
import random
from dataclasses import dataclass, field

@dataclass
class NetLayer:
    neuron_layer: list = field(default_factory=list) # List of Neurons

@dataclass
class Connections:
    weight: float = 0.0
    delta_weight: float = 0.0

class Neuron:

    eta: float = 0.15
    alpha: float = 0.5

    def __init__(self, num_out_connections:int, index:int):
        self.out_value: float
        self.gradient: float
        self.n_index = index
        self.out_weights: list = [] # List of Connections

        for conn in range(num_out_connections):
            conn = Connections()
            conn.weight = self._initial_weight()
            self.out_weights.append(conn) # List of Connections
            
    def _initial_weight(self):
        return random.random()

    def feed_forward(self, pre_layer: NetLayer):
        sum = 0
        for neuron in pre_layer.neuron_layer:
            sum += neuron.out_value * neuron.out_weights[self.n_index].weight

    def _transfer_func(self, x: float):
        return math.tanh(x)
        
    def _transfer_func_dx(self, x: float):
        return 1 - math.pow(x, 2)

    def get_out_gradients(self, target_value: float):
        delta = target_value - self.out_value
        self.gradient = delta * self._transfer_func_dx(self.out_value)

    def get_hidden_gradients(self, next_layer: NetLayer):
        sum = 0
        for x in range(len(next_layer.neuron_layer) - 1):
            sum += self.out_weights[x].weight * next_layer.neuron_layer[x].gradient
        self.gradient = sum * self._transfer_func_dx(self.out_value)
    
    def update_input_weights(self, pre_layer: NetLayer):
        for x in range(len(pre_layer.neuron_layer)):
            pre_neuron = pre_layer.neuron_layer[x]
            old_delta_weight = pre_neuron.out_weights[self.n_index].delta_weight
            new_delta_weight = eta * pre_neuron.out_value * self.gradient + alpha * old_delta_weight

            pre_neuron.out_weights[self.n_index].delta_weight = new_delta_weight
            pre_neuron.out_weights[self.n_index].weight += new_delta_weight
