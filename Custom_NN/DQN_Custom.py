import math
import random
import pickle
import os
from datetime import datetime
from copy import deepcopy
from PAWS_Bot_Navigation.Actions import Actions
from PAWS_Bot_Navigation.Network import Network
from PAWS_Bot_Navigation.Simulation import Simulation
from PAWS_Bot_Navigation.Plot import Plot
from PAWS_Bot_Navigation.Config import EPISODES, SIM_PORT, TIME_LIMIT


class DQN:

    def __init__(self, state_size: int, action_size: int, train_mode: bool):
        
        self.sim = Simulation()
        is_connected = self.sim.connect(SIM_PORT)
        if not is_connected:
            raise Exception("Not connected to remote API server")
        
        if train_mode:

            random.seed()

            num_hidden_layers = 3
            num_hidden_neuron = 12
            
            self.training_net = Network(state_size, action_size, num_hidden_layers, num_hidden_neuron)
            self.target_net = self._copy_net(self.training_net) # Needs to be a deep copy of training_net not reference

            self.experience = {} # state, action, reward, next_state
            self.memory = []

            self.memory_capacity = 500
            self.discount_rate = 0.95
            self.epsilon = 1.0
            self.epsilon_decay = 0.995
            self.epsilon_min = 0.01
            self.batch_size = 10
        

    def _get_predicted_action(self, state: list):
        # Returns action to take based on max Q value returned 
        # from prediction net
        if random.random() <= self.epsilon:
            return Actions(random.randint(0, len(Actions)-1))
        self.training_net.feed_forward(state)
        output = self.training_net.get_output()
        action_index = output.index(max(output))
        return Actions(action_index)
        
    def _get_target_value(self, next_state: list, reward: float):
        # Returns max Q value returned 
        # from target net
        self.target_net.feed_forward(next_state)
        target = self.target_net.get_output()
        max_q = float(max(target))
        return reward + self.discount_rate * max_q

    def _get_target_output(self, state: list, predicted_action: Actions, next_state: list, reward: float, done: bool):
        if done:
            target_value = reward
        else:
            target_value = self._get_target_value(next_state, reward)
        self.training_net.feed_forward(state)
        target_outputs = self.training_net.get_output()
        target_outputs[predicted_action.value] = target_value
        return target_outputs

    def _memorize(self, state, action, next_state, reward, done):
        experience = {
            "state": state,
            "action": action,
            "next_state": next_state,
            "reward": reward,
            "done": done        
        }
        if len(self.memory) < self.memory_capacity:
            self.memory.append(experience)
        else:
            self.memory.pop(0)
            self.memory.append(experience)

    def _get_memories(self):
        if len(self.memory) < self.batch_size:
            memories = []
        else: 
            memories = random.choices(self.memory, k=self.batch_size)
        return memories

    def _decay_epsilon(self):
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay      

    def _copy_net(self, net_to_copy):
        return deepcopy(net_to_copy)

    def _save_network(self, net_to_save: Network):
        now = datetime.now()
        now_str = now.strftime("%Y%m%d_%H%M%S")
        this_folder = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(
            this_folder, 
            f"saved_networks/net_{now_str}.pickle"
        )
        out_file = open(filepath, "wb")
        pickle.dump(net_to_save, out_file)
        out_file.close()

    def load_network(self, network_filepath: str) -> Network:
        self.training_net = pickle.load(open(network_filepath, "rb"))

    def get_action(self, state: list):
        # Returns action to take based on max Q value returned 
        # from prediction net
        self.training_net.feed_forward(state)
        output = self.training_net.get_output()
        action_index = output.index(max(output))
        return Actions(action_index)

    def train(self):
        done = False
        episode_plot = Plot("Time vs Episode")
        reward_plot = Plot("Average Reward vs Episode")

        for e in range(EPISODES):
            # Initialize environment
            print(f"EPISODE: {e} Initialized")
            err_plot = Plot("Error vs Time")
            self.sim.initialize()
            bot_init_position = self.sim.get_postion(self.sim.paws_bot)
            state = self.sim.get_state(bot_init_position)
            final_time = TIME_LIMIT
            average_reward = 0.0

            for time in range(TIME_LIMIT):
                # Get predicted action to advance the simulation
                print(time)
                predicted_action = self._get_predicted_action(state)
                next_state, reward, done = self.sim.step(
                    state,                    
                    predicted_action
                )
                
                average_reward += reward

                if len(next_state) > 0:
                    # Save current state in replay memory
                    self._memorize(
                        state, 
                        predicted_action,  
                        next_state, 
                        reward,
                        done
                    )

                if done:
                    # Update target net to training net weights
                    self.target_net = self._copy_net(self.training_net)
                    final_time = time
                    print(f"SUCCESS! - EPISODE: {e}, TIME: {time}")
                    break

                # Update state to next state
                state = next_state                

                # Use replay memory to train net
                memories = self._get_memories()

                for mem in memories:
                    state = mem["state"]
                    action = mem["action"]
                    next_state = mem["next_state"]
                    reward = mem["reward"]
                    done = mem["done"]

                    # Get target values
                    target_outputs = self._get_target_output(
                        state, 
                        action,
                        next_state,
                        reward,
                        done
                    )
                    
                    # Back propogate training net using target values
                    self.training_net.back_prop(target_outputs)
                
                err_plot.add_point(time, self.training_net.error_rms)
                if time % 10 == 0:
                    print(f"TIME: {time}, ERROR RMS: {self.training_net.error_rms}")

                # Reduce chance of exploration
                self._decay_epsilon()

            # Save plot at end of episode
            episode_plot.add_point(e, final_time)
            reward_plot.add_point(e, average_reward)

            now = datetime.now()
            now_str = now.strftime("%Y%m%d_%H%M%S")
            err_plot.plot(f"PAWS_Bot_Navigation/saved_data/plot_error_{e}_{now_str}")

        now = datetime.now()
        now_str = now.strftime("%Y%m%d_%H%M%S")
        episode_plot.plot(f"PAWS_Bot_Navigation/saved_data/plot_steps_{e}_{now_str}")
        reward_plot.plot(f"PAWS_Bot_Navigation/saved_data/plot_reward_{e}_{now_str}")

        # Save the trained net to use later
        self._save_network(self.training_net)

