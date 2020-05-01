from DQN import DQN
from Action import Action

class Simulation:

    def step(self, action: Action):
        pass # returns new_state, reward, done/goal, info

    def get_reward(self, state, done: bool):
        # Return the reward based on the reward policy
        # If done return goal reward
        pass 

    def reset(self):
        # Reset robot to start and pick random waypoint location
        pass

    def get_new_waypoint(self):
        pass

    def get_state(self):
        pass