from DQN_Code.Actions import Actions
import DQN_Code.CoppeliaSim as sim
import time

class Simulation:

    def __init__(self, sim_port: int):
        ''' Code required to connect to the Coppelia Remote API'''

        print ('Simulation started')
        sim.simxFinish(-1) # close all opened connections
        clientID = sim.simxStart('127.0.0.1', sim_port, True, True, 5000, 5) # Connect to CoppeliaSim
        if clientID!=-1:
            sim.simxAddStatusbarMessage(clientID, 'Hello! PAWS Connected.', sim.simx_opmode_oneshot)
            print ('Connected to remote API server')
        else:
            print ('Not connected to remote API server')

    def step(self, action: Actions):
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