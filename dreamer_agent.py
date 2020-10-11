import sys
import time
import argparse

import numpy as np
import torch

from donkeycar.parts.network import MQTTValuePub, MQTTValueSub

sys.path.insert(1, "/u/95/zhaoy13/unix/ICRA/donkeycar-dreamer")
from agent import Dreamer

parser = argparse.ArgumentParser()

# parser.add_argument("--car_name", help="Name of the car on MQTT-server", default="Kari")
parser.add_argument("--car_name", help="Name of the car on MQTT-server", default="dreamer")
parser.add_argument("--episode_steps", help="Number of steps per episode", default=1000, type=int)
parser.add_argument("--episodes", help="Number of steps episodes per run", default=100, type=int)

args = parser.parse_args()


class AttrDict(dict):
  __setattr__ = dict.__setitem__
  __getattr__ = dict.__getitem__

def define_config():
  args = AttrDict()
  args.belief_size = 200
  args.state_size = 30
  args.hidden_size = 300
  args.embedding_size = 1024
  args.observation_size = (3, 64, 64)
  args.action_size = 2
  args.device = 'cuda'
  args.testing_device = "cpu"
  args.symbolic = False
  args.dense_act = 'elu'
  args.cnn_act = 'relu'
  args.pcont = False
  args.pcont_scale = 10
  args.world_lr = 6e-4
  args.actor_lr = 8e-5
  args.value_lr = 8e-5
  args.free_nats = 3
  args.experience_size = 1000000
  args.bit_depth = 5
  args.discount = 0.99
  args.temp = 0.2 # entropy temperature
  args.disclam = 0.95
  args.planning_horizon =15
  args.batch_size =50
  args.chunk_size =50
  args.grad_clip_norm = 100.0
  args.with_logprob = False
  args.expl_amount = 0.3
  return args

DONKEY_NAME = args.car_name

STEER_LIMIT_LEFT = -1
STEER_LIMIT_RIGHT = 1
THROTTLE_MAX = 0.6
THROTTLE_MIN = 0.25
MAX_STEERING_DIFF = 0.25
STEP_LENGTH = 0.1
RANDOM_EPISODES = 1
GRADIENT_STEPS = 100

SKIP_INITIAL_STEPS = 20
MAX_EPISODE_STEPS = args.episode_steps + SKIP_INITIAL_STEPS

COMMAND_HISTORY_LENGTH = 5
FRAME_STACK = 1
VAE_OUTPUT = 20
LR = 0.0001

IMAGE_SIZE = 40
RGB = False

PARAMS = {

  "sac": {
    "linear_output": VAE_OUTPUT + COMMAND_HISTORY_LENGTH * 3,
    "lr": LR,
    "target_entropy": -2,
    "batch_size": 128,
    "hidden_size": 64,
    "encoder_update_frequency": 0,
    "critic_loss_encoder_update": True,
    "pretrained_ae": "",
    "im_size": IMAGE_SIZE,
    "n_images": 20000,
    "epochs": 1000
  },
  "ae": {
    "framestack": FRAME_STACK,
    "output": VAE_OUTPUT,
    "linear_input": 100,
    "image_size": IMAGE_SIZE,
    "lr": LR / 10,
    "image_channels": 3 if RGB else 1,
    "encoder_type": "vae",
    "batch_size": 64,
    "l2_regularization": False
  }
}


class RL_Agent():
  def __init__(self, alg_type, sim, car_name="Dreamer"):
    self.args = define_config()
    self.agent = Dreamer(self.args)
    self.sim = sim

    self.image = np.zeros((120, 160, 3))
    self.observation = torch.zeros((1, 3, 64, 64))  # init observation, with batch dim
    self.belief = torch.zeros(1, self.args.belief_size, device=self.args.device)
    self.posterior_state = torch.zeros(1, self.args.state_size, device=self.args.device)
    self.action = torch.zeros(1, self.args.action_size, device=self.args.device)

    self.speed = 0
    self.step = 0
    self.episode = 0
    self.episode_reward = 0
    self.replay_buffer = []
    self.target_speed = 0
    self.steering = 0

    self.training = False
    self.step_start = 0

    self.replay_buffer_pub = MQTTValuePub(car_name + "buffer", broker="mqtt.eclipse.org")
    self.replay_buffer_sub = MQTTValueSub(car_name + "buffer", broker="mqtt.eclipse.org")

    self.param_pub = MQTTValuePub(car_name + "param", broker="mqtt.eclipse.org")
    self.param_sub = MQTTValueSub(car_name + "param", broker="mqtt.eclipse.org")

    self.params = self.param_sub.run()

  def reset(self, observation):
    self.episode += 1

    self.episode_reward = 0
    self.replay_buffer = []

    self.target_speed = 0
    self.steering = 0

    self.observation = observation
    self.buffer_sent = False
    self.buffer_received = False
    self.params_sent = False
    self.params_received = False

  def train(self):
    # send buffer and receive new param
    # print(f"Training for {int(time.time() - self.training_start)} seconds")

    if not self.buffer_sent:
      print("Buffer sent")
      self.replay_buffer_pub.run(self.replay_buffer[SKIP_INITIAL_STEPS:])
      self.buffer_sent = True

    if (time.time() - self.training_start) > 60:
      """Temporary fix for when sometimes the replay buffer fails to send"""
      self.training_start = time.time()
      self.buffer_sent = False
      self.replay_buffer_pub.run(False)
      return False

    new_params = self.param_sub.run()

    if not new_params:
      return True

    print("Received new params.")
    self.replay_buffer_pub.run(False)
    self.agent.import_parameters(new_params)

    self.buffer_sent = False

    return False

  def run(self, image, speed=None):
    # TODO: what's the format of the image
    # called by rl_local_control.py, the most important function in this file.
    if not speed:
      self.speed = self.target_speed
    else:
      self.speed = speed

    if image is not None:
      self.image = image  # get observation, np with size (120, 160, 3)


    self.dead = self.is_dead(self.image) if not self.sim else self.is_dead_sim(self.image)

    if self.step == 0:
      if not self.sim:
        input("Press Enter to start a new episode.")
      self.reset(self.agent.process_im(self.image))

    if self.step > 0 and not self.training:
      """ get reward, done, next_obs (the img passed into) and save one-step traj to replay buffer"""
      reward = 1 + (self.speed - THROTTLE_MIN) / (THROTTLE_MAX - THROTTLE_MIN)
      done = self.dead
      reward = reward * -10 if self.dead else reward

      # get nex obs according to img passed in
      next_observation = self.agent.process_im(self.image)

      self.replay_buffer.append((self.observation,
                                 self.action.cpu(),
                                 reward,
                                 done))

      self.episode_reward += reward
      step_end = time.time()

      self.observation = next_observation

      print(
        f"Episode: {self.episode}, Step: {self.step}, Reward: {reward:.2f}, Episode reward: {self.episode_reward:.2f}, Step time: {(self.step_start - step_end):.2f}, Speed: {self.speed:.2f}")

    if self.step > MAX_EPISODE_STEPS or (self.dead and not self.training):
      # episode handler: prepare to train
      self.training_start = time.time()

      self.step = 0
      self.steering = 0
      self.target_speed = 0
      self.training = True

      return self.steering, self.target_speed, self.training

    if self.training:
      # begin to train, send collected data to GPU and train the models on GPU
      self.training = self.train()
      self.dead = False

      return self.steering, self.target_speed, self.training

    self.step += 1

    if self.step < SKIP_INITIAL_STEPS:
      # skip initial 20 steps to wait initialization
      return 0, 0, False

    self.step_start = time.time()

    # if self.episode < RANDOM_EPISODES:
    #    action = action_space.sample()
    # else:
    with torch.no_grad():
      self.belief, self.posterior_state = self.agent.infer_state(self.observation.to(self.args.device),
                                                                 action=self.action,
                                                                 belief=self.belief,
                                                                 state=self.posterior_state)
      self.action = self.agent.select_action((self.belief, self.posterior_state))  # TODO: change this

    # to get steering and target_speed as numpy
    action = self.action.cpu().numpy()
    self.steering, self.target_speed = action[0][0], action[0][1]  # action has batch dimension
    ## didn't use enforce_limit yet
    # self.steering, self.target_speed = self.enforce_limits(action, self.command_history[0]) # TODO: change this

    return self.steering, self.target_speed, self.training

  def is_dead(self, img):
    """
    Counts the black pixels from the ground and compares the amount to a threshold value.
    If there are not enough black pixels the car is assumed to be off the track.
    """

    crop_height = 20
    crop_width = 20
    threshold = 70
    pixels_percentage = 0.10

    pixels_required = (img.shape[1] - 2 * crop_width) * crop_height * pixels_percentage

    crop = img[-crop_height:, crop_width:-crop_width]

    r = crop[:, :, 0] < threshold
    g = crop[:, :, 1] < threshold
    b = crop[:, :, 2] < threshold

    pixels = (r & g & b).sum()

    # print("Pixels: {}, Required: {}".format(pixels, pixels_required))

    return pixels < pixels_required

  def is_dead_sim(self, img):

    crop_height = 40
    required = 0.8

    cropped = img[-crop_height:]

    rgb = cropped[:, :, 0] > cropped[:, :, 2]

    return rgb.sum() / (crop_height * 160) > required

  def enforce_limits(self, action, prev_steering):
    """
    Scale the agent actions to environment limits
    """

    var = (THROTTLE_MAX - THROTTLE_MIN) / 2
    mu = (THROTTLE_MAX + THROTTLE_MIN) / 2

    steering_min = max(STEER_LIMIT_LEFT, prev_steering - MAX_STEERING_DIFF)
    steering_max = min(STEER_LIMIT_RIGHT, prev_steering + MAX_STEERING_DIFF)

    steering = max(steering_min, min(steering_max, action[0]))

    return [steering, action[1] * var + mu]


if __name__ == "__main__":
  print("Starting as training server")
  agent = RL_Agent("dreamer", True, DONKEY_NAME)
  params_sent = False
  buffer_received = False
  trained = False
  training_episodes = 0

  while training_episodes < args.episodes:
    new_buffer = agent.replay_buffer_sub.run()

    if new_buffer and not trained:
      print(f"{len(new_buffer)} new buffer observations")
      agent.agent.append_buffer(new_buffer)
      print("Training")
      agent.agent.update_parameters(GRADIENT_STEPS)
      params = agent.agent.export_parameters()
      trained = True

    if trained:
      print("Sending parameters")
      agent.param_pub.run(params)
      time.sleep(1)

    if not new_buffer:
      agent.param_pub.run(False)
      trained = False
      print("Waiting for observations.")

    time.sleep(0.1)





