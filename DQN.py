import math
import random
import pickle
import os
import numpy as np
from datetime import datetime
from copy import deepcopy
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from PAWS_Bot_Navigation.Actions import Actions
from PAWS_Bot_Navigation.Simulation import Simulation
from PAWS_Bot_Navigation.Plot import Plot
from PAWS_Bot_Navigation.Config import (
    EPISODES, 
    SIM_PORT, 
    TIME_LIMIT, 
    ALPHA,
    MEMORY_CAPACITY,
    DISCOUNT_RATE,
    EPSILON,
    EPSILON_DECAY,
    EPSILON_MIN,
    BATCH_SIZE
)


class DQN:

    def __init__(self, state_size: int, action_size: int, train_mode: bool, model_filepath: str = ""):
        
        self.sim = Simulation()
        is_connected = self.sim.connect(SIM_PORT)
        
        if not is_connected:
            raise Exception("Not connected to remote API server")
        
        self.memory_capacity = MEMORY_CAPACITY
        self.discount_rate = DISCOUNT_RATE
        self.epsilon = EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_min = EPSILON_MIN
        self.batch_size = BATCH_SIZE
        self.state_size = state_size
        self.action_size = action_size

        self.training_net = self._create_model()

        if train_mode:
            random.seed()
            self.target_net = self._copy_net(self.training_net) # Needs to be a deep copy of training_net not reference
            self.experience = {} # state, action, reward, next_state
            self.memory = []
        else:
            self.load_weights(model_filepath)

    def _create_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.action_size))
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=ALPHA))
        return model

    def _get_predicted_action(self, state):
        # Returns action to take based on max Q value returned 
        # from prediction net
        if random.random() <= self.epsilon:
            return Actions(random.randint(0, len(Actions)-1))
        output = self.training_net.predict(state.reshape(1,-1))[0]
        action_index = np.argmax(output)
        return Actions(action_index)
        
    def _get_target_value(self, next_state, reward: float):
        # Returns max Q value returned 
        # from target net
        target = self.target_net.predict(next_state.reshape(1,-1))[0]
        max_q = np.amax(target)
        return reward + self.discount_rate * max_q

    def _get_target_output(self, state, predicted_action: Actions, next_state, reward: float, done: bool):
        if done:
            target_value = reward
        else:
            target_value = self._get_target_value(next_state, reward)
        target_outputs = self.training_net.predict(state.reshape(1,-1))[0]
        old_output = target_outputs[predicted_action.value] 
        target_outputs[predicted_action.value] = target_value
        return target_outputs, old_output

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

    def _save_network(self, net_to_save):
        now = datetime.now()
        now_str = now.strftime("%Y%m%d_%H%M%S")
        this_folder = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(
            this_folder, 
            f"saved_networks/net_{now_str}.h5"
        )
        net_to_save.save_weights(filepath)

    def load_weights(self, net_weights_filepath: str):
        self.training_net.load_weights(net_weights_filepath)
        
    def get_action(self, state):
        # Returns action to take based on max Q value returned 
        # from prediction net
        output = self.training_net.predict(state.reshape(1,-1))[0]
        action_index = output.index(max(output))
        return Actions(action_index)
    
    def _replay(self):
        # Use replay memory to train net
        print("Replay Training...")
        memories = self._get_memories()
        
        avg_error_rms = 0.0
        sum_d = 0.0

        for mem in memories:
            replay_state = mem["state"]
            replay_action = mem["action"]
            replay_next_state = mem["next_state"]
            replay_reward = mem["reward"]
            replay_done = mem["done"]

            # Get target values
            target_outputs, old_output = self._get_target_output(
                replay_state, 
                replay_action,
                replay_next_state,
                replay_reward,
                replay_done
            )
            
            # Back propogate training net using target values
            self.training_net.fit(replay_state.reshape(1,-1), target_outputs.reshape(1,-1), epochs=1, verbose=0)

            delta = target_outputs[replay_action.value] - old_output
            sum_d += math.pow(delta, 2)
        
        if len(memories) > 0:
            avg_error_rms = math.sqrt(sum_d/float(len(memories)))
        
        print("Done")

        return avg_error_rms   

    def test(self):
        self.sim.initialize()
        
        bot_init_position = self.sim.get_postion(self.sim.paws_bot)
        state = self.sim.get_state(bot_init_position)
        final_time = TIME_LIMIT

        for time in range(TIME_LIMIT):
                # Get predicted action to advance the simulation
                predicted_action = self._get_predicted_action(state)
                
                next_state, _, done = self.sim.step(
                    state,                    
                    predicted_action
                )

                if time % 10 == 0:
                    dist = self.sim.get_length([next_state[4], next_state[5]])
                    print(f"TIME: {time}, DISTANCE FROM GOAL: {dist}")

                if done:                    
                    final_time = time
                    if len(next_state) == 0:
                        print("AN ERROR OCURRED")
                    else:
                        print(f"REACHED GOAL! TIME: {time}")
                    break
                
                # Update state to next state
                state = next_state


    def train(self):
        done = False
        steps_plot = Plot("Time vs Episode")
        reward_plot = Plot("Average Reward vs Episode")

        for e in range(EPISODES):
            # Initialize environment
            print(f"EPISODE: {e} Initialized")
            err_plot = Plot("Error vs Time")

            self.sim.initialize()
            bot_init_position = self.sim.get_postion(self.sim.paws_bot)
            state = self.sim.get_state(bot_init_position)
            final_time = TIME_LIMIT
            cumu_reward = 0.0

            for time in range(TIME_LIMIT):
                # Get predicted action to advance the simulation
                predicted_action = self._get_predicted_action(state)
                
                next_state, reward, done = self.sim.step(
                    state,                    
                    predicted_action
                )
                
                cumu_reward += reward

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
                    if len(next_state) == 0:
                        print("AN ERROR OCURRED")
                    else:
                        print(f"REACHED GOAL! - EPISODE: {e}, TIME: {time}")
                    break

                batch_error_rms = self._replay()
                
                # Update state to next state
                state = next_state 

                err_plot.add_point(time, batch_error_rms)
                
                if time % 10 == 0:
                    dist = self.sim.get_length([next_state[4], next_state[5]])
                    print(f"TIME: {time}, STEP_ERROR: {batch_error_rms}, DISTANCE FROM GOAL: {dist}")

                # Reduce chance of exploration
                self._decay_epsilon()

            # Save plot at end of episode
            steps_plot.add_point(e, final_time)
            reward_plot.add_point(e, cumu_reward/final_time)

            now = datetime.now()
            now_str = now.strftime("%Y%m%d_%H%M%S")
            err_plot.plot(f"PAWS_Bot_Navigation/saved_data/plot_error_{e}_{now_str}")

        # Save plots at end of training
        now = datetime.now()
        now_str = now.strftime("%Y%m%d_%H%M%S")
        steps_plot.plot(f"PAWS_Bot_Navigation/saved_data/plot_steps_{e}_{now_str}")
        reward_plot.plot(f"PAWS_Bot_Navigation/saved_data/plot_reward_{e}_{now_str}")

        # Save the trained net to use later
        self._save_network(self.training_net)

