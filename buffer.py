import numpy as np
import torch
from env import postprocess_observation, preprocess_observation_


class ExperienceReplay():
  def __init__(self, size, symbolic_env, observation_size, action_size, bit_depth, device, env=None):
    # TODO: change the number of size: from env_size --> game_size
    self.env = env
    self.device = device
    self.symbolic_env = symbolic_env
    self.full = False
    self.bit_depth = bit_depth

    self.observation_size = observation_size
    self.action_size = action_size
    self.size = size//1000  # buffer size: how many games the buffer can store

    self.game_idx = 0
    self.num_game, self.num_steps = 0, 0  # to track the total amount of games and env_steps

    self.buffer = {}  # used to store games
    # init the buffer of one game
    self.buffer[self.game_idx] = {"obs": [], "action": [], "reward": [], "nonterminal": []}

  def append(self, observation, action, reward, done):
    # append data: if the game is not finished, append it; if a game is finished, append the game to self.buffer
    if self.symbolic_env:
      self.buffer[self.game_idx]["obs"].append(observation.numpy())
    else:
      self.buffer[self.game_idx]["obs"].append(postprocess_observation(observation.numpy(), self.bit_depth))
    self.buffer[self.game_idx]["action"].append(action.numpy())
    self.buffer[self.game_idx]["reward"].append(reward)
    self.buffer[self.game_idx]["nonterminal"].append(not done)
    # print(self.buffer[self.game_idx]["nonterminal"])
    self.num_steps += 1

    if done:
      # when a game is finished, change the idx of game and init new game_buffer
      self.game_idx = (self.game_idx + 1) % self.size  # when buffer is full, filling buffer begins from head
      self.full = self.full or self.game_idx == 0
      self.buffer[self.game_idx] = {"obs": [], "action": [], "reward": [], "nonterminal": []}
      self.num_game += 1

  # Enable to append data beyond the terminal state
  def _sample_game_idx(self, batch_size):
    # sample a valid index of each batch
    game_upper_range = len(self.buffer)-1  # -1 is because when done, we init a new game in the buffer
    # print("game_upper", game_upper_range)
    return [np.random.randint(0, game_upper_range) for _ in range(batch_size)]

  def _retrieve_game(self, game_idx, chunk_size):
    game = self.buffer[game_idx]
    # print(game)
    game_length = len(game["nonterminal"])
    position_idx = np.random.randint(0, game_length)  # sample position

    _observations = np.empty((chunk_size, self.observation_size) if self.symbolic_env else (chunk_size, 3, 64, 64), dtype=np.float32 if self.symbolic_env else np.uint8)
    _actions = np.empty((chunk_size, self.action_size), dtype=np.float32)
    _rewards = np.empty((chunk_size, ), dtype=np.float32)
    _nonterminals = np.empty((chunk_size, ), dtype=np.float32)

    # fill data, when position is beyond the boundary, use generated data
    for i, position in enumerate(range(position_idx, position_idx + chunk_size)):
      if position < game_length:
        _observations[i] = game["obs"][position]
        _actions[i] = game["action"][position]
        _rewards[i] = game["reward"][position]
        _nonterminals[i] = game["nonterminal"][position]
      else:
        # fill data beyond terminal state
        _observations[i] = game["obs"][-1]
        if self.env:
          _actions[i] = self.env.sample_random_action()
        else:
          _actions[i] = game["action"][-1]
        _rewards[i] = 0
        _nonterminals[i] = game["nonterminal"][-1]  # always done
    return [_observations, _actions, _rewards, _nonterminals]

  def _retrieve_batch(self, batch_size, chunk_size):
    sampled_game_idx = self._sample_game_idx(batch_size)

    observations = []
    actions = []
    rewards = []
    nonterminals = []

    for idx in sampled_game_idx:
      _observations, _actions, _rewards, _nonterminals = self._retrieve_game(idx, chunk_size)
      observations.append(_observations)
      actions.append(_actions)
      rewards.append(_rewards)
      nonterminals.append(_nonterminals)

    observations = torch.as_tensor(np.array(observations, dtype=np.float32))
    if not self.symbolic_env:
      preprocess_observation_(observations, self.bit_depth)  # Undo discretisation for visual observations

    observations = observations.reshape(chunk_size, batch_size, *observations.shape[-3:])
    actions = np.array(actions).reshape(chunk_size, batch_size, -1)
    rewards = np.array(rewards).reshape(chunk_size, batch_size)
    nonterminals = np.array(nonterminals).reshape(chunk_size, batch_size)

    return observations, actions, rewards, nonterminals

  def sample(self, batch_size, chunk_size):
    """
    sample from buffer
    :param n: batch-size
    :param L: chunk-size
    :return: an array with shape batch-size * chunk-size where each item includes obs, act, rewards, nonterminal_flag
    """
    batch = self._retrieve_batch(batch_size, chunk_size)
    return [torch.as_tensor(item).to(device=self.device) for item in batch]
