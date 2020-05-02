import math
import random
import pickle
from datetime import datetime as dt
from copy import deepcopy
from DQN_Code.Actions import Actions
from DQN_Code.Network import Network
from DQN_Code.Simulation import Simulation

EPISODES = 500

class DQN:

    def __init__(self, state_size: int, action_size: int, train_mode: bool):
        
        self.sim = Simulation()
        
        if train_mode:

            random.seed()

            num_hidden_layers = 3
            num_hidden_neuron = 8
            
            self.training_net = Network(state_size, action_size, num_hidden_layers, num_hidden_neuron)
            self.target_net = self._copy_net(self.training_net) # Needs to be a deep copy of training_net not reference

            self.experience: dict # state, action, reward, next_state
            self.memory: list

            self.memory_capacity = 1000
            self.discount_rate = 0.95
            self.epsilon = 1.0
            self.epsilon_decy = 0.95
            self.epsilon_min = 0.01
            self.learning_rate = 0.001
            self.time_limit = 500
            self.batch_size = 10
        

    def _get_predicted_action(self, state: list):
        # Returns action to take based on max Q value returned 
        # from prediction net
        if random.random() <= self.epsilon:
            return Actions[random.randint(0, len(Actions))]
        output = self.training_net.feed_forward(state)
        action_index = output.index(max(output))
        return Actions[action_index]
        
    def _get_target_value(self, next_state: list, reward: int):
        # Returns max Q value returned 
        # from target net
        target = self.target_net.feed_forward(next_state)
        max_q = max(target)
        return reward + self.learning_rate * max_q

    def _get_target_output(self, state: list, next_state: list, reward: int):
        target_outputs = self.target_net.feed_forward(state)
        target_value = self._get_target_value(next_state, reward)
        target_outputs[predicted_action] = target_value
        return target_outputs

    def _memorize(self, state, action, reward, next_state, done):
        experience = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done        
        }
        if len(self.memory) < self.memory_capacity:
            self.memory.append(experience)
        else:
            self.memory.pop(0)
            self.memory.append(experience)

    def _get_memories(self):
        if len(memory) == 0:
            return {}
        return random.choices(self.memory, k=self.batch_size)

    def _decay_epsilon(self):
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decy

    def _copy_net(self, net_to_copy):
        return deepcopy(net_to_copy)

    def _save_network(self, net_to_save: Network):
        filepath = f"saved_networks/net.{dt.today()}.pickle"
        out_file = open(filepath, "wb")
        pickle.dump(net_to_save, out_file)
        out_file.close()

    def load_network(self, network_filepath: str) -> Network:
        self.training_net = pickle.load(open(network_filepath, "rb"))

    def get_action(self, state: list):
        # Returns action to take based on max Q value returned 
        # from prediction net
        output = self.training_net.feed_forward(state)
        action_index = output.index(max(output))
        return Actions[action_index]

    def train(self):
        sim_done = False

        for e in range(EPISODES):
            # Initialize environment
            print(f"EPISODE: {e} Initialized")
            self.sim.reset()
            sim_state = self.sim.get_state()

            for time in range(self.time_limit):
                # Get predicted action to advance the simulation
                sim_action = self._get_predicted_action(sim_state)
                sim_next_state, sim_reward, sim_done = sim.step(
                    predicted_action
                )

                # Update state to next state
                sim_state = sim_next_state
                
                # Save current state in replay memory
                self._memorize(
                    sim_state, 
                    sim_action, 
                    sim_reward, 
                    sim_next_state, 
                    sim_done
                )

                if sim_done:
                    print(f"SUCCESS! - EPISODE: {e}, TIME: {time}, REWARD: {sim.reward}")
                    break

                # Use replay memory to train net
                memories = self._get_memories()

                for mem in memories:
                    state = mem["state"]
                    next_state = mem["next_state"]
                    reward = mem["reward"]
                    done = mem["done"]

                    # Get target values
                    target_outputs = self._get_target_output(
                        state, 
                        next_state,
                        reward
                    )
                    
                    # Back propogate training net using target values
                    self.training_net.back_prop(target_outputs)

                if time % 10 == 0:
                    print(f"ERROR RMS: {self.training_net.error_rms}")
                    
                    # Update target net to training net weights
                    self.target_net = self._copy_net(self.training_net)

                # Reduce chance of exploration
                self._decay_epsilon()

        # Save the trained net to use later
        self._save_network(self.training_net)
