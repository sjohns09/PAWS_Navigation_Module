import math
import random
from dataclasses import dataclass, field
from PAWS_Bot_Navigation.Config import ALPHA, ETA

@dataclass
class NetLayer:
    neuron_layer: list = field(default_factory=list) # List of Neurons

@dataclass
class Connections:
    weight: float = 0.0
    delta_weight: float = 0.0

class Neuron:

    def __init__(self, num_out_connections:int, index:int):
        self.out_value = 0.0
        self.gradient = 0.0
        self.n_index = index
        self.out_weights: list = [] # List of Connections

        for conn in range(num_out_connections):
            conn = Connections()
            conn.weight = self._initial_weight()
            self.out_weights.append(conn) # List of Connections
            
    def _initial_weight(self):
        return random.random()

    def feed_forward(self, pre_layer: NetLayer):
        sum_w = 0.0
        for neuron in pre_layer.neuron_layer:
            sum_w += neuron.out_value * neuron.out_weights[self.n_index].weight
        self.out_value = self._transfer_func(sum_w)

    def _transfer_func(self, x: float):
        # Relu activation function
        return max(0.0, x)
        
    def _transfer_func_dx(self, x: float):
        if x > 0:
            slope = 1.0
        else:
            slope = 0.0
        return slope

    def get_out_gradients(self, target_value: float):
        # This is where the loss function is implemented
        # Loss for DQN = (Target - Prediction)^2
        # gradient = learning_rate*[Qtarget - currentQ]*gradient(currentQ)
        delta = target_value - self.out_value
        self.gradient = ALPHA * delta * self._transfer_func_dx(self.out_value)
        print(f"gradient {self.gradient}")

    def get_hidden_gradients(self, next_layer: NetLayer):
        sum_w = 0.0
        for x in range(len(next_layer.neuron_layer) - 1):
            sum_w += self.out_weights[x].weight * next_layer.neuron_layer[x].gradient
        self.gradient = sum_w * self._transfer_func_dx(self.out_value)
    
    def update_input_weights(self, pre_layer: NetLayer):
        for x in range(len(pre_layer.neuron_layer)):
            pre_neuron = pre_layer.neuron_layer[x]
            old_delta_weight = pre_neuron.out_weights[self.n_index].delta_weight
            new_delta_weight = ETA * pre_neuron.out_value * self.gradient + ALPHA * old_delta_weight

            pre_neuron.out_weights[self.n_index].delta_weight = new_delta_weight
            pre_neuron.out_weights[self.n_index].weight += new_delta_weight
