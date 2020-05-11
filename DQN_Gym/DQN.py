import math
import random
import os
import numpy as np
import gym
import gym_maze
from datetime import datetime
from copy import deepcopy
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from PAWS_Bot_Navigation.Utilities.Plot import Plot
from PAWS_Bot_Navigation.DQN_Gym.Config import (
    EPISODES, 
    SIM_PORT, 
    TIME_LIMIT, 
    ALPHA,
    MEMORY_CAPACITY,
    DISCOUNT_RATE,
    EPSILON,
    EPSILON_DECAY,
    EPSILON_MIN,
    BATCH_SIZE,
    PLOT_SAVE_FOLDER,
    TARGET_UPDATE_COUNT,
    NETWORK_SAVE_FOLDER,
    GYM_ENV,
    REWARD_FREE_SPACE,
    REWARD_DISTANCE_WEIGHT,
    REWARD_CLOSE_WEIGHT,
    REWARD_TIME_DECAY,
    GOAL_REWARD,
    NOT_SAFE_REWARD
)


class DQN:

    def __init__(self, state_size: int, action_size: int, train_mode: bool, model_filepath: str = ""):
        
        # Initialize the "maze" environment
        print(f"Making maze environemt {GYM_ENV}")
        self.env = gym.make(GYM_ENV)
        # Number of discrete states (bucket) per state dimension

        self.num_grids = tuple((self.env.observation_space.high + np.ones(self.env.observation_space.shape)).astype(int))

        # Bounds for each discrete state
        self.state_bounds = list(zip(self.env.observation_space.low, self.env.observation_space.high))
        self.goal_state = np.array([self.state_bounds[0][-1], self.state_bounds[1][-1]])
        self.optimal_distance = np.linalg.norm(self.goal_state)

        self.env.render()
        
        self.this_folder = os.path.dirname(os.path.abspath(__file__))

        self.epsilon = EPSILON
        self.state_size = state_size
        self.action_size = self.env.action_space.n

        self.training_net = self._create_model()

        if train_mode:
            random.seed()
            self.target_net = self._create_model() # Needs to be a deep copy of training_net not reference
            self._update_target_weights()
            self.experience = {} # state, action, reward, next_state, done
            self.memory = []
        else:
            self._load_weights(model_filepath)

    def train(self):
        update_target = 0

        # Setup Episode Stats
        now = datetime.now()
        now_str = now.strftime("%Y%m%d_%H%M%S")
        steps_plot = Plot("Time vs Episode")
        reward_plot = Plot("Average Reward vs Episode")
        sim_reward_plot = Plot("Average Simulation Reward vs Episode")
        err_plot = Plot("Error vs Episode")

        for e in range(EPISODES):
            # Initialize environment
            print(f"EPISODE: {e} Initialized")         

            obs = self.env.reset()

            final_time = TIME_LIMIT
            cumu_reward = []
            cumu_sim_reward = []
            batch_error_rms = []
            done = False
            reward = 0.0

            for time in range(TIME_LIMIT):
                if time == 0:
                    state = self._get_state(obs)

                # Get predicted action to advance the simulation
                predicted_action = self._get_predicted_action(state)
                next_obs, reward_sim, done, info = self.env.step(predicted_action)
                next_state = self._get_state(next_obs)

                reward = self._get_reward(state, next_state, done, time)

                # Stat tracking
                cumu_reward.append(reward)
                cumu_sim_reward.append(reward_sim)
                
                # Save current state in replay memory
                self._memorize(
                    state, 
                    predicted_action,  
                    next_state, 
                    reward,
                    done
                )

                # Determine when to update target net
                update_target += 1
                if update_target > TARGET_UPDATE_COUNT:
                    self._update_target_weights()
                    update_target = 0

                if done:
                    final_time = time
                    print(f"REACHED GOAL! - EPISODE: {e}, AVG_REWARD: {np.mean(cumu_reward)}, TIME: {time}")
                    break

                # Train NN
                error = self._replay()
                if len(self.memory) >= BATCH_SIZE:
                    batch_error_rms.append(error)
                
                # Reduce chance of exploration
                self._decay_epsilon()

                # Update state to next state
                state = next_state 

            # Stat Tracking
            print(f"EPISODE {e}, AVG_REWARD: {np.mean(cumu_reward)}, AVG_SIM_REWARD: {np.mean(cumu_sim_reward)}")
            steps_plot.add_point(e, final_time)
            reward_plot.add_point(e, np.mean(cumu_reward))
            sim_reward_plot.add_point(e, np.mean(cumu_sim_reward))
            err_plot.add_point(e, np.mean(batch_error_rms))

            if e % 100 == 0:
                # Save intermittent network and plots for partial training
                print(f"Saving Network and Plots for episode {e}")
                self._save_network(self.training_net, e)
                steps_plot.plot(os.path.join(self.this_folder, f"{PLOT_SAVE_FOLDER}", f"plot_steps_{e}_{now_str}"))
                reward_plot.plot(os.path.join(self.this_folder, f"{PLOT_SAVE_FOLDER}", f"plot_reward_{e}_{now_str}"))
                sim_reward_plot.plot(os.path.join(self.this_folder, f"{PLOT_SAVE_FOLDER}", f"plot_sim_reward_{e}_{now_str}"))
                err_plot.plot(os.path.join(self.this_folder, f"{PLOT_SAVE_FOLDER}", f"plot_error_{e}_{now_str}"))

        # Save plots at end of training
        steps_plot.plot(os.path.join(self.this_folder, f"{PLOT_SAVE_FOLDER}", f"plot_steps_{e}_{now_str}"))
        reward_plot.plot(os.path.join(self.this_folder, f"{PLOT_SAVE_FOLDER}", f"plot_reward_{e}_{now_str}"))
        sim_reward_plot.plot(os.path.join(self.this_folder, f"{PLOT_SAVE_FOLDER}", f"plot_sim_reward_{e}_{now_str}"))
        err_plot.plot(os.path.join(self.this_folder, f"{PLOT_SAVE_FOLDER}", f"plot_error_{e}_{now_str}"))

        # Save the trained net to use later
        self._save_network(self.training_net, EPISODES)

    def test(self, now_str):
        done = False
        obs = self.env.reset()
        final_time = TIME_LIMIT

        # recording_folder = PLOT_SAVE_FOLDER
        # self.env.monitor().start(recording_folder, force=True)

        for time in range(TIME_LIMIT):
            if time == 0:
                state = self._get_state(obs)

            # Get predicted action to advance the simulation
            predicted_action = self._get_predicted_action(state)
            next_obs, reward, done, info = self.env.step(predicted_action)
            
            # Update state to next state
            next_state = self._get_state(next_obs)
            state = next_state

            if len(next_state) == 0:
                print("AN ERROR OCURRED")
                break

            if done:                    
                final_time = time
                print(f"REACHED GOAL! TIME: {time}")
                break
        
        # self.env.monitor.close()

        return done, final_time

    def _get_reward(self, state, next_state, done: bool, time):
        # Return the reward based on the reward policy
        reward = 0.0
        old_dist = np.linalg.norm([self.goal_state - state])
        new_dist = np.linalg.norm([self.goal_state - next_state])
        if done:
            reward = GOAL_REWARD # goal reward
        elif np.array_equal(state, next_state):
            reward = NOT_SAFE_REWARD
        else:
            # Positive reward for moving towards the objective
            # Negative for moving away
            r_distance = old_dist - new_dist
            
            # Higher reward for being near the objective
            r_close = (self.optimal_distance-new_dist)*REWARD_FREE_SPACE
            
            # Reward decays the longer time has passed
            r_time = (time/TIME_LIMIT) * REWARD_TIME_DECAY

            # Total reward
            reward = REWARD_DISTANCE_WEIGHT*r_distance + REWARD_CLOSE_WEIGHT*r_close + r_time
        return reward

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
            return random.randint(0, self.action_size-1)
        output = self.training_net.predict(state.reshape(1,-1))[0]
        return int(np.argmax(output))
        
    def _get_target_value(self, next_state, reward: float):
        # Returns max Q value returned 
        # from target net
        target = self.target_net.predict(next_state.reshape(1,-1))[0]
        max_q = np.amax(target)
        return reward + DISCOUNT_RATE * max_q

    def _get_target_output(self, state, predicted_action, next_state, reward: float, done: bool):
        if done:
            target_value = reward
        else:
            target_value = self._get_target_value(next_state, reward)
        target_outputs = self.training_net.predict(state.reshape(1,-1))[0]
        old_output = target_outputs[predicted_action] 
        target_outputs[predicted_action] = target_value
        return target_outputs, old_output

    def _memorize(self, state, action, next_state, reward, done):
        experience = {
            "state": state,
            "action": action,
            "next_state": next_state,
            "reward": reward,
            "done": done        
        }
        if len(self.memory) < MEMORY_CAPACITY:
            self.memory.append(experience)
        else:
            self.memory.pop(0)
            self.memory.append(experience)

    def _get_memories(self):
        if len(self.memory) < BATCH_SIZE:
            memories = []
        else: 
            memories = random.choices(self.memory, k=BATCH_SIZE)
        return memories

    def _decay_epsilon(self):
        if self.epsilon >= EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY      

    def _update_target_weights(self):
        self.target_net.set_weights(self.training_net.get_weights())

    def _save_network(self, net_to_save, ep_id):
        now = datetime.now()
        now_str = now.strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(
            self.this_folder, 
            NETWORK_SAVE_FOLDER,
            f"net_{now_str}_e{ep_id}.h5"
        )
        net_to_save.save_weights(filepath)

    def _load_weights(self, net_weights_filepath: str):
        self.training_net.load_weights(net_weights_filepath)
        
    def _get_action(self, state):
        # Returns action to take based on max Q value returned 
        # from trained net
        output = self.training_net.predict(state.reshape(1,-1))[0]
        action_index = np.argmax(output)
        return action_index
    
    def _replay(self):
        # Use replay memory to train net
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

            delta = target_outputs[replay_action] - old_output
            sum_d += math.pow(delta, 2)
        
        if len(memories) > 0:
            avg_error_rms = math.sqrt(sum_d/float(len(memories)))

        return avg_error_rms   

    def _get_state(self, obs):
        grid_indice = []
        for i in range(len(obs)):
            if obs[i] <= self.state_bounds[i][0]:
                grid_index = 0
            elif obs[i] >= self.state_bounds[i][1]:
                grid_index = self.num_grids[i] - 1
            else:
                # Mapping the state bounds to the grid array
                bound_width = self.state_bounds[i][1] - self.state_bounds[i][0]
                offset = (self.num_grids[i]-1)*self.state_bounds[i][0]/bound_width
                scaling = (self.num_grids[i]-1)/bound_width
                grid_index = int(round(scaling*obs[i] - offset))
            grid_indice.append(grid_index)
        return np.array([grid_indice])
