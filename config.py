# Simulation Settings
PAWS_BOT_MODEL_PATH = 'D:\Sammie\Documents\Grad School\Spring 2020 - RobotLearning\Project\PAWS_Bot_Navigation\CoppeliaSim\paws_bot_model.ttm'
SIM_PORT = 19999

# Robot Settings
MAX_MOTOR_SPEED = 40
MAX_TURN_SPEED = 7
MIN_MOTOR_SPEED = 0
COLLISION_DIST = 0.05 # meters
STEP_DISTANCE = 0.45 # meters per move
TOLERANCE = 0.45 # meters to goal

# Environment Settings
STATE_SIZE = 6
ACTION_SIZE = 4

# Reward Settings
GOAL_REWARD = 10.0
NOT_SAFE_REWARD = -5.0
REWARD_DISTANCE_WEIGHT = 5

# Training Settings
EPISODES = 20
TIME_LIMIT = 150
MEMORY_CAPACITY = 500
BATCH_SIZE = 10

# Hyperparameters
ALPHA = 0.01 # Learning Rate
DISCOUNT_RATE = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.95
EPSILON_MIN = 0.01


