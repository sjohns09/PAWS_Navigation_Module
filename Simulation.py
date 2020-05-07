import time
import random
import math
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
    STEP_DISTANCE
)

class Simulation:

    def __init__(self):
        random.seed()

        # Set Floor Size min max (square)
        self.floor_points = (-5, 5)

        # Possible Locations for Human
        self.waypoint_array = self.float_range(-4.5, -3.5, 0.1)
        self.waypoint_array.extend(self.float_range(3.5, 4.5, 0.1)) 
        
        self.initial_pos = [0, 0]
        self.deg_180 = 180
        self.deg_90 = 90

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
        return math.sqrt(sum(v**2 for v in vector))
    
    def get_vector(self, coords_1, coords_2):
        # Assumes planar vector
        return [coords_2[0]-coords_1[0], coords_2[1]-coords_1[1]]
        
    def step(self, old_state, action: Actions):
        done = False
        is_safe = self.move(action)
        new_position = self.get_postion(self.paws_bot)
        new_state = self.get_state(new_position)

        if self.get_length([new_state[4], new_state[5]]) <= TOLERANCE:
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
            self._turn_left(self.deg_180), 
            is_safe = self._move_forward(STEP_DISTANCE)
        elif action == Actions.LEFT:
            self._turn_left(self.deg_90), 
            is_safe = self._move_forward(STEP_DISTANCE)
        elif action == Actions.RIGHT:
            self._turn_right(self.deg_90), 
            is_safe = self._move_forward(STEP_DISTANCE)
        
        return is_safe
        
    def _set_target_velocity(self, motor, speed):
        sim.simxSetJointTargetVelocity(
            self.client_id, 
            motor, 
            speed, 
            sim.simx_opmode_streaming
        )

    def _move_forward(self, target_dist):
        is_safe = True
        start_position = self.get_postion(self.paws_bot, -1)
        current_position = start_position
        
        self._set_target_velocity(self.paws_right_motor, MAX_MOTOR_SPEED)
        self._set_target_velocity(self.paws_left_motor, MAX_MOTOR_SPEED)
        
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
            self.paws_west_sensor
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
            r_distance = (old_dist-current_dist)*2.0
            # Better reward for finding optimal path
            r_optimal = math.exp(-current_dist/self.optimal_distance)
            # Total reward
            reward = 1 + r_distance + r_optimal
        return reward

            



        