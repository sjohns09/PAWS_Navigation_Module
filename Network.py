import math
from dataclasses import dataclass
from PAWS_Bot_Navigation.Neuron import (
    Connections,
    NetLayer,
    Neuron
)


class Network:

    def __init__(self, num_inputs: int, num_outputs: int, num_hidden_layers: int, num_hidden_neuron: int):
        self.num_layers = num_hidden_layers + 2
        self.output_num = num_outputs + 1
        self.input_num = num_inputs + 1
        self.hidden_num = num_hidden_neuron + 1
        
        self.error_rms: float = 0.0
        self.net_layers: list # List of NetLayer

        num_out_connections = 0
        for layer_num in range(0, self.num_layers):
            self.net_layers.append(NetLayer())

            if layer_num < self.num_layers - 2:
                num_out_connections = self.hidden_num
            elif layer_num == self.num_layers - 2:
                num_out_connections = self.output_num
            else:
                num_out_connections = 0

            if layer_num != self.num_layers - 1 and layer_num != 0:
                for i in range(self.hidden_num):
                    self.net_layers[layer_num].neuron_layer.append(Neuron(num_out_connections, i))
            elif layer_num == 0:
                for i in range(self.input_num):
                    self.net_layers[layer_num].neuron_layer.append(Neuron(num_out_connections, i))
            else:
                for i in range(self.output_num):
                    self.net_layers[layer_num].neuron_layer.append(Neuron(num_out_connections, i))

            self.net_layers[layer_num].neuron_layer[-1].out_value = 1

    def feed_forward(self, inputs: list):
        i = 0
        for input in inputs:
            self.net_layers[0].neuron_layer[i].out_value = input
            i += 1

        for layer_num in range(1, self.num_layers):
            pre_layer = self.net_layers[layer_num - 1]
            for n in range(len(self.net_layers[layer_num]) - 1):
                self.net_layers[layer_num].neuron_layer[n].feed_forward(pre_layer)


    def back_prop(self, targets: list): # target values should be determined from target Q network
        output_layer = self.net_layers[-1]
        self.error_rms = 0.0

        # Get RMS error
        for n in range(self.output_num - 1):
            delta = targets[n] - output_layer[n].out_value
            self.errorRMS += math.pow(delta, 2)
        self.error_rms = math.sqrt((1 / (self.output_num - 1)) * self.error_rms)
        
        # Get output layer gradients
        for n in range(self.output_num - 1):
            output_layer.neuron_layer[n].get_out_gradients(targets[n])

        # Get hidden layer gradients
        right_hidden_index = self.num_layers - 2
        for l in range(right_hidden_index, 0, -1):
            for n in range(self.hidden_num):
                self.net_layers[l].neuron_layer[n].get_hidden_gradients(self.net_layers[l + 1])

        # Update Connection weights
        for l in range(self.num_layers - 1, 0, -1):
            for n in range(len(self.net_layers[l]) - 1):
                self.net_layers[l].neuron_layer[n].update_input_weights(self.net_layers[l - 1])

    def get_output(self):
        results = []
        for i in range(self.output_num - 1):
            results.append(self.net_layers[-1].neuron_layer[i].out_value)

        return results        
