from PAWS_Bot_Navigation.Actions import Actions
import PAWS_Bot_Navigation.CoppeliaSim.CoppeliaSim as sim
from PAWS_Bot_Navigation.config import PAWS_BOT_MODEL_PATH
import time
import random

class Simulation:

    def __init__(self):
        random.seed()

        # Set Floor Size
        self.floor_points = (-5, 5)

        # Possible Locations for Human
        self.waypoint_array = self.float_range(-5, -3, 0.1)
        self.waypoint_array.extend(self.float_range(3, 5, 0.1)) 

        self.tol = 0.01

    def float_range(self, x, y, step):
        range = list()
        while x <= y:
            range.append(x)
            x += step
            x = round(x, 2)
        return range

    def connect(self, sim_port: int):
        # Code required to connect to the Coppelia Remote API
        print ('Simulation started')
        sim.simxFinish(-1) # close all opened connections
        self.client_id = sim.simxStart('127.0.0.1', sim_port, True, True, 5000, 5) # Connect to CoppeliaSim
        if self.client_id!=-1:
            sim.simxAddStatusbarMessage(
                self.client_id, 'Hello! PAWS Connected.', sim.simx_opmode_oneshot
            )
            print ('Connected to remote API server')
            is_connected = True
        else:
            print ('Not connected to remote API server')
            is_connected = False

        if is_connected:
            # Get necessary object handles
            self._get_object_handles()
            
        return is_connected
    
    def _get_object_handles(self):
        err_code, self.paws_bot = sim.simxGetObjectHandle(
            self.client_id, 
            "paws_bot", 
            sim.simx_opmode_blocking
        )
        err_code, self.paws_front_sensor = sim.simxGetObjectHandle(
            self.client_id, 
            "paws_northProxSensor", 
            sim.simx_opmode_blocking
        )
        err_code, self.paws_back_sensor = sim.simxGetObjectHandle(
            self.client_id, 
            "paws_southProxSensor", 
            sim.simx_opmode_blocking
        )
        err_code, self.paws_right_sensor = sim.simxGetObjectHandle(
            self.client_id, 
            "paws_eastProxSensor", 
            sim.simx_opmode_blocking
        )
        err_code, self.paws_left_sensor = sim.simxGetObjectHandle(
            self.client_id, 
            "paws_westProxSensor", 
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

    def distance(self, coords_1, coords_2):
        return math.sqrt(
            (coords_2[0]-coords_1[0])**2 + (coords_2[1]-coords_1[1])**2
        )

    def step(self, old_state, action: Actions):
        done = false
        self.move(self.paws_bot, action)
        new_position = self.get_postion(self.paws_bot)
        new_state = self.get_state(new_position)

        if self.distance(new_position, self.human_coords) <= self.tol:
            done = true
        
        reward = self._get_reward(old_state, new_state, done)

        return new_state, reward, done

    def get_postion(self, object_handle: int, relative_to: int = -1):
        # relative_to = -1 is the absolute position
        return sim.simxGetObjectPosition(
            self.client_id,
            object_handle,
            relative_to,
            sim.simx_opmode_blocking
        )

    def get_state(self, position):
        # Needs to return free or occupied for N S E W
        # and distance between bot and human
        pass

    def move(self, action: Actions):
        sim.simxSetJointTargetVelocity(self.client_id, self.paws_left_motor, 10, sim.simx_opmode_streaming)
        sim.simxSetJointTargetVelocity(self.client_id, self.paws_right_motor, 10, sim.simx_opmode_streaming)


    def reset(self):
        # Reset robot to start
        absolute_ref = -1
        
        sim.simxRemoveModel(
            self.client_id, 
            self.paws_bot, 
            sim.simx_opmode_blocking
        )
        sim.simxLoadModel(
            self.client_id, 
            PAWS_BOT_MODEL_PATH, 
            0, 
            sim.simx_opmode_blocking
        )
        self._get_object_handles()

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
        # If done return goal reward = 1
        # If move into obstacle (collision) = 0
        pass 