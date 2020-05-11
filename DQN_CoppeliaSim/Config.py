# Simulation Settings
PAWS_BOT_MODEL_PATH = "D:\Sammie\Documents\Grad School\Spring 2020 - RobotLearning\Project\PAWS_Bot_Navigation\CoppeliaSim\paws_bot_model.ttm"
SIM_PORT = 19999
PLOT_SAVE_FOLDER = 'saved_data'
NETWORK_SAVE_FOLDER = 'saved_networks'

# Test Settings
NUM_TEST_RUNS = 5

# Robot Settings
MAX_MOTOR_SPEED = 40
MAX_TURN_SPEED = 5
MIN_MOTOR_SPEED = 0
COLLISION_DIST = 0.10 # meters
STEP_DISTANCE = 0.50 # meters per move
TOLERANCE = 0.50 # meters to goal

# Environment Settings
STATE_SIZE = 7
ACTION_SIZE = 4

# Reward Settings
GOAL_REWARD = 10
NOT_SAFE_REWARD = -1
REWARD_DISTANCE_WEIGHT = 0.4
REWARD_CLOSE_WEIGHT = 0.6
REWARD_FREE_SPACE = 1
REWARD_TIME_DECAY = -0.5 # Max penalty for taking too long

# Training Settings
EPISODES = 500
TIME_LIMIT = 50
MEMORY_CAPACITY = 1000
BATCH_SIZE = 35
TARGET_UPDATE_COUNT = 25

# Hyperparameters
ALPHA = 0.05 # Learning Rate
DISCOUNT_RATE = 0.95
EPSILON = 1
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01


