from enum import Enum


class EnvConfig(Enum):
	car = "dreamer"
	server = "mqtt.eclipse.org"
	bit_depth = 5
	STEER_LIMIT_LEFT = -1
	STEER_LIMIT_RIGHT = 1
	THROTTLE_MAX = 0.6
	THROTTLE_MIN = 0.25
	MAX_STEERING_DIFF = 0.2
	STEP_LENGTH = 0.1

	MAX_EPISODE_STEPS = 500

	COMMAND_HISTORY_LENGTH = 5
	FRAME_STACK = 1
	VAE_OUTPUT = 20

	LR_START = 0.0001
	LR_END = 0.0001
	ANNEAL_END_EPISODE = 50

	IMAGE_SIZE = 40
	RGB = False

