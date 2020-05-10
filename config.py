# Simulation Settings
PAWS_BOT_MODEL_PATH = 'CoppeliaSim\paws_bot_model.ttm'
SIM_PORT = 19999
PLOT_SAVE_FOLDER = 'saved_data'
NETWORK_SAVE_FOLDER = 'saved_networks'

# Test Settings
NUM_TEST_RUNS = 10

# Robot Settings
MAX_MOTOR_SPEED = 20
MAX_TURN_SPEED = 3
MIN_MOTOR_SPEED = 0
COLLISION_DIST = 0.1 # meters
STEP_DISTANCE = 0.50 # meters per move
TOLERANCE = 0.75 # meters to goal

# Environment Settings
STATE_SIZE = 7
ACTION_SIZE = 4

# Reward Settings
GOAL_REWARD = 100
NOT_SAFE_REWARD = -2.5
REWARD_DISTANCE_WEIGHT = 0.4
REWARD_CLOSE_WEIGHT = 0.6
REWARD_FREE_SPACE = 1
REWARD_TIME_DECAY = -0.5 # Max penalty for taking too long

# Training Settings
EPISODES = 500
TIME_LIMIT = 100
MEMORY_CAPACITY = 1000
BATCH_SIZE = 35
TARGET_UPDATE_COUNT = 25

# Hyperparameters
ALPHA = 0.01 # Learning Rate
DISCOUNT_RATE = 0.90
EPSILON = 1.0
EPSILON_DECAY = 0.99
EPSILON_MIN = 0.01


