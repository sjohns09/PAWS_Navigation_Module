from PAWS_Bot_Navigation.Actions import Actions
import PAWS_Bot_Navigation.CoppeliaSim as sim
import time
import random
import numpy as np

class Simulation:

    def __init__(self):
        random.seed()

        # Set Floor Size
        self.floor_points = (-5, 5)

        # Possible Locations for Human
        self.waypoint_array = np.linspace(-5, -3, num=100)
        self.waypoint_array = np.append(
            self.waypoint_array, 
            np.linspace(5, 3, num=100)
        )

        self.tol = 0.005

    def connect(self, sim_port: int):
        # Code required to connect to the Coppelia Remote API
        print ('Simulation started')
        sim.simxFinish(-1) # close all opened connections
        self.client_id = sim.simxStart('localhost', sim_port, True, True, 5000, 5) # Connect to CoppeliaSim
        if self.client_id!=-1:
            sim.simxAddStatusbarMessage(
                self.client_id, 'Hello! PAWS Connected.', sim.simx_opmode_oneshot
            )
            print ('Connected to remote API server')
        else:
            print ('Not connected to remote API server')

        # Get necessary object handles
        err_code, self.paws_bot = sim.simxGetObjectHandle(
            self.client_id, 
            "paws_bot", 
            sim.simx_opmode_blocking
        )
        err_code, self.paws_front_sensor = sim.simxGetObjectHandle(
            self.client_id, 
            "paws_nProximitySensor", 
            sim.simx_opmode_blocking
        )
        err_code, self.paws_left_motor = sim.simxGetObjectHandle(
            self.client_id, 
            "paws_leftMotor", 
            sim.simx_opmode_blocking
        )
        err_code, self.paws_right_motor = sim.simxGetObjectHandle(
            self.client_id, 
            "paws_rightMotor", 
            sim.simx_opmode_blocking
        )
        err_code, self.human = sim.simxGetObjectHandle(
            self.client_id,
            "human", 
            sim.simx_opmode_blocking
        )
        
    def step(self, action: Actions):
        done = false
        current_state = self.get_state(self.paws_bot)
        self.move(action)
        new_state = self.get_state(self.paws_bot)

        if (new_state - self.human_coords) <= self.tol:
            done = true
        reward = self._get_reward(new_state)

        pass # returns new_state, reward, done

    def get_state(self, object_handle: int, relative_to: int = -1):
        # relative_to = -1 is the absolute position
        return sim.simxGetObjectPosition(
            self.client_id,
            object_handle,
            relative_to,
            sim.simx_opmode_blocking
        )
    def move(self, action: Actions):
        pass

    def reset(self):
        # Reset robot to start
        absolute_ref = -1
        zero_position = (0, 0, 0)
        sim.simxSetObjectPosition(
            self.client_id, 
            self.paws_bot, 
            absolute_ref, 
            zero_position, 
            sim.simx_opmode_blocking
        )
        # Set random waypoint location for human on edge of map
        self.human_coords = self._get_random_location(self.waypoint_array)
        err = sim.simxSetObjectPosition(
            self.client_id, 
            self.human, 
            absolute_ref, 
            (
                self.human_coords[0], 
                self.human_coords[1], 
                self.human_coords[2]
            ),
            sim.simx_opmode_blocking
        )

    def close_connection(self):
        sim.simxFinish(-1)
    
    def _get_random_location(self, array_of_points):
        # Returns X, Y, Z location using random points in array
        i_x = random.randint(0, len(array_of_points)-1)
        i_y = random.randint(0, len(array_of_points)-1)
        z = 0 # Planar 
        return (array_of_points[i_x], array_of_points[i_y], z)

    def _get_reward(self, state, done: bool):
        # Return the reward based on the reward policy
        # If done return goal reward
        pass 