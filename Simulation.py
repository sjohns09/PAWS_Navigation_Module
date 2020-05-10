import time
import random
import math
import numpy as np
import os
from more_itertools import locate
from scipy.spatial.transform import Rotation
from scipy.spatial import distance as dist
from PAWS_Bot_Navigation.Actions import Actions
from PAWS_Bot_Navigation.CoppeliaSim import CoppeliaSim as sim
from PAWS_Bot_Navigation.Config import ( 
    PAWS_BOT_MODEL_PATH,
    MAX_MOTOR_SPEED,
    MIN_MOTOR_SPEED,
    MAX_TURN_SPEED,
    COLLISION_DIST,
    TOLERANCE,
    STEP_DISTANCE,
    GOAL_REWARD,
    NOT_SAFE_REWARD,
    REWARD_DISTANCE_WEIGHT,
    REWARD_CLOSE_WEIGHT,
    REWARD_TIME_DECAY,
    TIME_LIMIT,
    REWARD_FREE_SPACE
)

class Simulation:

    def __init__(self):
        random.seed()
        self.this_folder = os.path.dirname(os.path.abspath(__file__))

        # Set Floor Size min max (square)
        self.floor_points = (-5, 5)

        # Possible Locations for Human
        array = np.linspace(-4, -2.5, num=15)
        self.waypoint_array = np.concatenate((array, np.linspace(2.5, 4, num=15))) 
        
        self.initial_pos = np.array([0, 0, 0])
        self.deg_90 = 90

    def connect(self, sim_port: int):
        # Code required to connect to the Coppelia Remote API
        print ('Simulation started')
        self.sim_port = sim_port
        sim.simxFinish(-1) # close all opened connections
        self.client_id = sim.simxStart('127.0.0.1', sim_port, True, False, 5000, 3) # Connect to CoppeliaSim
        if self.client_id!=-1:
            self.display_info('Hello! PAWS Connected.')
            print ('Connected to remote API server')
            is_connected = True
        else:
            print ('Not connected to remote API server')
            is_connected = False

        if is_connected:
            # Get necessary object handles
            self._get_object_handles()
            
        return is_connected
    
    def display_info(self, info):
        sim.simxAddStatusbarMessage(
                self.client_id, info, sim.simx_opmode_oneshot
            )
    
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
        )[1]
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
        
    def set_optimal_distance(self, initial_pos, final_pos):
        vector = self.get_vector(initial_pos, final_pos)
        self.optimal_distance = self.get_length(vector)

    def get_length(self, vector):
        return float(np.linalg.norm(vector))
    
    def get_vector(self, coords_1, coords_2):
        # Assumes planar vector
        x = coords_2[0]-coords_1[0]
        y = coords_2[1]-coords_1[1]
        return np.array([x, y])
        
    def step(self, old_state, action: Actions, time, old_waypoint_dist):
        done = False
        is_safe = self.move(action)
        new_position = self.get_postion(self.paws_bot)
        
        # If it goes off the map, end the Episode
        if any(list(map(lambda coord: abs(coord) > self.floor_points[1], new_position))):
            new_state = []
            reward = NOT_SAFE_REWARD
            return new_state, reward, done
        
        new_state, waypoint_dist = self.get_state(new_position, time)

        if waypoint_dist <= TOLERANCE:
            done = True
        
        reward = self._get_reward(old_waypoint_dist, waypoint_dist, is_safe, done, time)

        return new_state, reward, done, waypoint_dist

    def get_postion(self, object_handle: int, relative_to: int = -1):
        # relative_to = -1 is the absolute position
        pos = sim.simxGetObjectPosition(
            self.client_id,
            object_handle,
            relative_to,
            sim.simx_opmode_blocking
        )[1]
        return np.array(pos)

    def get_state(self, bot_position, time):
        # free or occupied for N S E W sensor readings
        sensor_raw = self._get_sensor_readings()
        #sensor_bools = np.fromiter(map(lambda x: int(x == True), sensor_raw), bool)
        # normalized vector between bot and human
        vector = self.get_vector(bot_position, self.human_coords)
        len_vector = self.get_length(vector) # Need for future calculations
        norm_vector = vector/len_vector
        # length of time spent in episode
        norm_time = time/TIME_LIMIT
        return np.concatenate((sensor_raw, norm_vector, [norm_time])), len_vector

    def _get_sensor_readings(self):
        sensor_list = [
            self.paws_north_sensor,
            self.paws_south_sensor,
            self.paws_east_sensor,
            self.paws_west_sensor
        ]
        detection_list = []
        for sensor in sensor_list:
            detection_list.append(
                sim.simxReadProximitySensor(
                    self.client_id, 
                    sensor, 
                    sim.simx_opmode_blocking
                    )[1]
            )
        
        return detection_list

    def move(self, action: Actions):
        # Need to write move function
        # needs to return a collision indicator (success, fail)
        # Turn in direction given by action
        is_safe = True

        if action == Actions.FORWARD:
            is_safe = self._move_forward(STEP_DISTANCE)
        elif action == Actions.BACKWARD: 
            is_safe = self._move_forward(STEP_DISTANCE, reverse=True)
        elif action == Actions.LEFT:
            self._turn_left(self.deg_90)
            is_safe = self._move_forward(STEP_DISTANCE)
        elif action == Actions.RIGHT:
            self._turn_right(self.deg_90)
            is_safe = self._move_forward(STEP_DISTANCE)
        
        return is_safe
        
    def _set_target_velocity(self, motor, speed):
        sim.simxSetJointTargetVelocity(
            self.client_id, 
            motor, 
            speed, 
            sim.simx_opmode_streaming
        )

    def _move_forward(self, target_dist, reverse=False):
        is_safe = True
        start_position = self.get_postion(self.paws_bot, -1)
        current_position = start_position
        
        max_speed = MAX_MOTOR_SPEED
        if reverse == True:
            max_speed = -MAX_MOTOR_SPEED
        self._set_target_velocity(self.paws_right_motor, max_speed)
        self._set_target_velocity(self.paws_left_motor, max_speed)
        
        condition = True
        while (condition):
            current_position = self.get_postion(self.paws_bot, -1)
            vector = self.get_vector(start_position, current_position)
            dist = self.get_length(vector)
            condition = dist < target_dist
            if self._detect_collision():
                is_safe = False
                print('COLLISION')
                break

        self._set_target_velocity(self.paws_right_motor, MIN_MOTOR_SPEED)
        self._set_target_velocity(self.paws_left_motor, MIN_MOTOR_SPEED)

        return is_safe

    def _detect_collision(self):
        collision = False
        det_state = []
        det_dist = []
        sensor_list = [
            self.paws_north_sensor,
            self.paws_east_sensor,
            self.paws_west_sensor,
            self.paws_south_sensor
        ]
        for sensor in sensor_list:
            det_reading = sim.simxReadProximitySensor(
                    self.client_id, 
                    sensor, 
                    sim.simx_opmode_blocking
                )
            det_state.append(det_reading[1])
            det_dist.append(det_reading[2])
            
        index = list(locate(det_state, lambda det: det == True))
        for i in index:
            dist = self.get_length(det_dist[i])
            if dist <= COLLISION_DIST:
                collision = True
                break
        return collision

    def _get_object_rot(self, object_id):
        start_quaternion = sim.simxGetObjectQuaternion(
            self.client_id, 
            object_id, 
            -1, 
            sim.simx_opmode_blocking
        )[1]
        
        if all(q == 0 for q in start_quaternion):
            rot_deg = 0.0
            is_increasing = 0
        else:
            r = Rotation.from_quat(start_quaternion)
            rot_vec = r.as_euler('zyx', degrees=True)
            rot_deg = rot_vec[1]
            is_increasing = rot_vec[0] >= 0

        return rot_deg, is_increasing

    def _turn_left(self, turn_rad):
        object_turn_cumulative = 0.0
        turn_deg, is_increasing = self._get_object_rot(self.paws_bot)

        self._set_target_velocity(self.paws_right_motor, MAX_TURN_SPEED)
        self._set_target_velocity(self.paws_left_motor, -MAX_TURN_SPEED)
        
        self._turn_control(turn_rad, turn_deg, is_increasing)

        self._set_target_velocity(self.paws_right_motor, MIN_MOTOR_SPEED)
        self._set_target_velocity(self.paws_left_motor, MIN_MOTOR_SPEED)

    def _turn_right(self, turn_rad):
        
        turn_deg, is_increasing = self._get_object_rot(self.paws_bot)

        self._set_target_velocity(self.paws_left_motor, MAX_TURN_SPEED)
        self._set_target_velocity(self.paws_right_motor, -MAX_TURN_SPEED)

        self._turn_control(turn_rad, turn_deg, is_increasing)
        
        self._set_target_velocity(self.paws_left_motor, MIN_MOTOR_SPEED)
        self._set_target_velocity(self.paws_right_motor, MIN_MOTOR_SPEED)

    def _turn_control(self, turn_rad, initial_turn_deg, initial_is_increasing):
        object_turn_cumulative = 0.0

        while (object_turn_cumulative < turn_rad):
            step_turn_deg, step_is_increasing = self._get_object_rot(self.paws_bot)
            
            if ((initial_is_increasing >= 0) == (step_is_increasing >= 0)):
                object_turn_cumulative += abs(step_turn_deg - initial_turn_deg)
            else:
                turn_inc = abs(initial_turn_deg - 90)
                step_turn_inc = abs(step_turn_deg - 90)
                object_turn_cumulative += (turn_inc + step_turn_inc)

            initial_turn_deg = step_turn_deg
            initial_is_increasing = step_is_increasing
    
    def initialize(self):
        # Reset robot to start
        absolute_ref = -1
        model_path = os.path.join(self.this_folder, f"{PAWS_BOT_MODEL_PATH}")
        sim.simxRemoveModel(
            self.client_id, 
            self.paws_bot, 
            sim.simx_opmode_blocking
        )
        sim.simxLoadModel(
            self.client_id, 
            model_path, 
            0, 
            sim.simx_opmode_blocking
        )
        self._get_object_handles()

        # Set random waypoint location for human on edge of map
        self.human_coords = self._get_random_location(self.waypoint_array)
        sim.simxSetObjectPosition(
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
        return [array_of_points[i_x], array_of_points[i_y], z]

    def _get_reward(self, old_waypoint_dist, waypoint_dist, is_safe: bool, done: bool, time):
        # Return the reward based on the reward policy
        reward = 0.0
        if done:
            reward = GOAL_REWARD # goal reward
        elif not is_safe:
            reward = NOT_SAFE_REWARD # obstacle negative reward
        else:
            # Positive reward for moving towards the objective
            # Negative for moving away
            r_distance = old_waypoint_dist - waypoint_dist
            
            # Higher reward for being near the objective
            r_close = (self.optimal_distance-waypoint_dist)*REWARD_FREE_SPACE
            
            # Reward decays the longer time has passed
            r_time = (time/TIME_LIMIT) * REWARD_TIME_DECAY

            # Total reward
            reward = REWARD_DISTANCE_WEIGHT*r_distance + REWARD_CLOSE_WEIGHT*r_close + r_time
        return reward

            



        