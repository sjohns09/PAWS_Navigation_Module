import time
import random
import math
from scipy.spatial.transform import Rotation as rot
from scipy.spatial import distance as dist
from PAWS_Bot_Navigation.Actions import Actions
from PAWS_Bot_Navigation.CoppeliaSim import CoppeliaSim as sim
from PAWS_Bot_Navigation.config import ( 
    PAWS_BOT_MODEL_PATH,
    MAX_MOTOR_SPEED,
    MIN_MOTOR_SPEED,
    MAX_TURN_SPEED,
    COLLISION_DIST
)

class Simulation:

    def __init__(self):
        random.seed()

        # Set Floor Size min max (square)
        self.floor_points = (-5, 5)

        # Possible Locations for Human
        self.waypoint_array = self.float_range(-5, -4, 0.1)
        self.waypoint_array.extend(self.float_range(4, 5, 0.1)) 
        
        self.initial_pos = [0, 0]
        self.tol = 0.01 # meters
        self.deg_180 = 180
        self.deg_90 = 90
        self.step_distance = 0.5 # meters

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
        self.paws_bot = sim.simxGetObjectHandle(
            self.client_id, 
            "paws_bot", 
            sim.simx_opmode_blocking
        )[1]
        self.paws_north_sensor = sim.simxGetObjectHandle(
            self.client_id, 
            "paws_northProxSensor", 
            sim.simx_opmode_blocking
        )[1]
        self.paws_south_sensor = sim.simxGetObjectHandle(
            self.client_id, 
            "paws_southProxSensor", 
            sim.simx_opmode_blocking
        )[1]
        self.paws_east_sensor = sim.simxGetObjectHandle(
            self.client_id, 
            "paws_eastProxSensor", 
            sim.simx_opmode_blocking
        )[1]
        self.paws_west_sensor = sim.simxGetObjectHandle(
            self.client_id, 
            "paws_westProxSensor", 
            sim.simx_opmode_blocking
        )
        self.paws_left_motor = sim.simxGetObjectHandle(
            self.client_id, 
            "paws_leftMotor", 
            sim.simx_opmode_blocking
        )[1]
        self.paws_right_motor = sim.simxGetObjectHandle(
            self.client_id, 
            "paws_rightMotor", 
            sim.simx_opmode_blocking
        )[1]
        self.human = sim.simxGetObjectHandle(
            self.client_id,
            "human", 
            sim.simx_opmode_blocking
        )[1]
        
    def set_optimal_distance(self, inital_pos, final_pos):
        vector = self.get_vector(initial_pos, final_pos)
        self.optimal_distance = self.get_length(vector)

    def get_length(self, vector):
        return math.sqrt((vector[0])**2 + (vector[1])**2)
    
    def get_vector(self, coords_1, coords_2):
        # Assumes planar vector
        return [coords_2[0]-coords_1[0], coords_2[1]-coords_1[1]]
        
    def step(self, old_state, action: Actions):
        done = false
        self.move(self.paws_bot, action)
        is_safe = self.is_safe()
        new_position = self.get_postion(self.paws_bot)
        new_state = self.get_state(new_position)

        if self.get_length(new_state[4], new_state[5]) <= self.tol:
            done = true
        
        reward = self._get_reward(old_state, new_state, is_safe, done)

        return new_state, reward, done

    def get_postion(self, object_handle: int, relative_to: int = -1):
        # relative_to = -1 is the absolute position
        return sim.simxGetObjectPosition(
            self.client_id,
            object_handle,
            relative_to,
            sim.simx_opmode_blocking
        )[1]

    def get_state(self, bot_position):
        # Needs to return free or occupied for N S E W
        # and vector between bot and human
        n_ps, s_ps, e_ps, w_ps = self._get_sensor_readings()
        state = list(map(lambda x: int(x == True), [n_ps, s_ps, e_ps, w_ps]))
        vector = self.get_vector(bot_position, self.human_coords)
        state.extend(vector)
        return state

    def _get_sensor_readings(self):
        north_detection = sim.simxReadProximitySensor(
            self.client_id, 
            self.paws_north_sensor, 
            sim.simx_opmode_blocking
        )[1]
        south_detection = sim.simxReadProximitySensor(
            self.client_id, 
            self.paws_south_sensor, 
            sim.simx_opmode_blocking
        )[1]
        east_detection = sim.simxReadProximitySensor(
            self.client_id, 
            self.paws_east_sensor, 
            sim.simx_opmode_blocking
        )[1]
        west_detection = sim.simxReadProximitySensor(
            self.client_id, 
            self.paws_west_sensor, 
            sim.simx_opmode_blocking
        )[1]

        return (north_detection, south_detection, east_detection, west_detection)


    def move(self, action: Actions):
        # Need to write move function
        # needs to return a collision indicator (success, fail)
        # Turn in direction given by action
        
        if action == Actions.FORWARD:
            self._move_forward(self.step_distance)
        elif action == Actions.BACKWARD: 
            self._turn_left(self.deg_180), 
            self._move_forward(self.step_distance)
        elif action == Actions.LEFT:
            self._turn_left(self.deg_90), 
            self._move_forward(self.step_distance)
        elif action == Actions.RIGHT:
            self._turn_right(self.deg_90), 
            self._move_forward(self.step_distance)
    
    def _set_target_velocity(self, motor, speed):
        sim.simxSetJointTargetVelocity(
            self.client_id, 
            motor, 
            speed, 
            sim.simx_opmode_streaming
        )

    def _move_forward(self, target_dist):
        start_position = self.get_postion(self.paws_bot, -1)
        current_position = self.get_postion(self.paws_bot, -1)
        
        self._set_target_velocity(self.paws_right_motor, MAX_MOTOR_SPEED)
        self._set_target_velocity(self.paws_left_motor, MAX_MOTOR_SPEED)
        
        collision = 0
        while (dist.euclidean(current_position, start_position) < target_dist):
            current_position = self.get_postion(self.paws_bot, -1)
            
            # Check if collision
            north_det_dist = sim.simxReadProximitySensor(
                self.client_id, 
                self.paws_north_sensor, 
                sim.simx_opmode_blocking
            )[2]
            obj_dist = dist.euclidean(north_det_dist, (0,0,0))
            if obj_dist < COLLISION_DIST:
                break

        self._set_target_velocity(self.paws_right_motor, MIN_MOTOR_SPEED)
        self._set_target_velocity(self.paws_left_motor, MIN_MOTOR_SPEED)
        
    def _get_object_z_rot(self, object_id):
        start_quaternion = sim.simxGetObjectQuaternion(
            self.client_id, 
            object_id, 
            -1, 
            sim.simx_opmode_blocking
        )[1]
        
        if all(q == 0 for q in start_quaternion):
            z_rot = 0.0
        else:
            r = rot.from_quat(start_quaternion)
            rot_vec = r.as_euler('zyx', degrees=True)
            print(rot_vec)
        return rot_vec[2]

    def _turn_left(self, turn_rad):
        object_z_start = self._get_object_z_rot(self.paws_bot)
        object_z = object_z_start

        self._set_target_velocity(self.paws_right_motor, MAX_TURN_SPEED)
        self._set_target_velocity(self.paws_left_motor, -MAX_TURN_SPEED)
        
        while (abs(object_z_start - object_z) < turn_rad):
            object_z = self._get_object_z_rot(self.paws_bot)
        
        self._set_target_velocity(self.paws_right_motor, MIN_MOTOR_SPEED)
        self._set_target_velocity(self.paws_left_motor, MIN_MOTOR_SPEED)

    def _turn_right(self, turn_rad):
        object_z_start = self._get_object_z_rot(self.paws_bot)
        object_z = object_z_start

        self._set_target_velocity(self.paws_left_motor, MAX_TURN_SPEED)
        self._set_target_velocity(self.paws_right_motor, -MAX_TURN_SPEED)

        while (abs(object_z_start - object_z) < turn_rad):
            object_z = self._get_object_z_rot(self.paws_bot)
        
        self._set_target_velocity(self.paws_left_motor, MIN_MOTOR_SPEED)
        self._set_target_velocity(self.paws_right_motor, MIN_MOTOR_SPEED)

    def is_safe(self):
        # Determine if collided with object by checking sensors
        # might need distance returned after all
        return True
    
    def initialize(self):
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

        self.set_optimal_distance(self.initial_pos, self.human_coords)

    def close_connection(self):
        sim.simxFinish(-1)
    
    def _get_random_location(self, array_of_points):
        # Returns X, Y, Z location using random points in array
        i_x = random.randint(0, len(array_of_points)-1)
        i_y = random.randint(0, len(array_of_points)-1)
        z = 0 # Planar 
        return (array_of_points[i_x], array_of_points[i_y], z)

    def _get_reward(self, old_state, current_state, is_safe: bool, done: bool):
        # Return the reward based on the reward policy
        # If done return goal reward
        # If move into obstacle (collision) return bad reward
        reward = 0
        if done:
            reward = 10 # goal reward
        elif not is_safe:
            reward = -5 # obstacle negative reward
        else:
            # Better reward for moving towards the objective
            old_dist = self.get_length([old_state[4], old_state[5]])
            current_dist = self.get_length([current_state[4], current_state[5]])
            r_distance = current_dist-old_dist
            # Better reward for finding optimal path
            r_optimal = math.exp(-current_dist/self.optimal_distance)
            # Total reward
            reward = 1 + r_distance + r_optimal
        return reward

            



        