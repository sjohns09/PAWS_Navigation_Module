# Simulation Settings
PAWS_BOT_MODEL_PATH = 'D:\Sammie\Documents\Grad School\Spring 2020 - RobotLearning\Project\PAWS_Bot_Navigation\CoppeliaSim\paws_bot_model.ttm'
SIM_PORT = 19999
PLOT_SAVE_FOLDER = 'saved_data'
NETWORK_SAVE_FOLDER = 'saved_networks'

# Robot Settings
MAX_MOTOR_SPEED = 40
MAX_TURN_SPEED = 5
MIN_MOTOR_SPEED = 0
COLLISION_DIST = 0.05 # meters
STEP_DISTANCE = 0.45 # meters per move
TOLERANCE = 0.75 # meters to goal

# Environment Settings
STATE_SIZE = 6
ACTION_SIZE = 4

# Reward Settings
GOAL_REWARD = 10.0
NOT_SAFE_REWARD = -2.0
REWARD_DISTANCE_WEIGHT = 0.6
REWARD_CLOSE_WEIGHT = 0.4
REWARD_TIME_DECAY = 0.05

# Training Settings
EPISODES = 50
TIME_LIMIT = 100
MEMORY_CAPACITY = 1000
BATCH_SIZE = 20
TARGET_UPDATE_COUNT = 25

# Hyperparameters
ALPHA = 0.01 # Learning Rate
DISCOUNT_RATE = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.90
EPSILON_MIN = 0.01


