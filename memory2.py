import numpy as np
import torch
from env import postprocess_observation, preprocess_observation_


class ExperienceReplay():
  def __init__(self, size, symbolic_env, observation_size, action_size, bit_depth, device):
    self.device = device
    self.symbolic_env = symbolic_env
    self.size = size
    self.observations = np.empty((size, observation_size) if symbolic_env else (size, 1, 40, 40), dtype=np.float32)
    self.actions = np.empty((size, action_size), dtype=np.float32)
    self.rewards = np.empty((size, ), dtype=np.float32)
    # self.nonterminals = np.empty((size, 1), dtype=np.float32)
    self.nonterminals = np.empty((size, ), dtype=np.float32)
    self.idx = 0
    self.full = False  # Tracks if memory has been filled/all slots are valid
    self.steps, self.episodes = 0, 0  # Tracks how much experience has been used in total
    self.bit_depth = bit_depth
    self.ends_idx = []  # to store the idx of ends_steps of each episode

  def append(self, observation, action, reward, done):
    self.observations[self.idx] = observation.numpy()
    # if self.symbolic_env:
    #   self.observations[self.idx] = observation.numpy()
    # else:
    #   self.observations[self.idx] = postprocess_observation(observation.numpy(), self.bit_depth)  # Decentre and discretise visual observations (to save memory)
    self.actions[self.idx] = action.numpy() if isinstance(action, torch.Tensor) else action
    self.rewards[self.idx] = reward
    self.nonterminals[self.idx] = not done
    if done:
      self.ends_idx.append(self.idx)
    self.idx = (self.idx + 1) % self.size
    self.full = self.full or self.idx == 0
    self.steps, self.episodes = self.steps + 1, self.episodes + (1 if done else 0)

  # Returns an index for a valid single sequence chunk uniformly sampled from the memory
  def _sample_idx(self, L, need_ends=False):

    valid_idx = False
    while not valid_idx:
      if not need_ends:
        idx = np.random.randint(0, self.size if self.full else self.idx - L)
      else:  # draw samples with ends state
        idx = np.random.choice(self.ends_idx) - L + 1
      idxs = np.arange(idx, idx + L) % self.size
      valid_idx = not self.idx in idxs[1:]  # Make sure data does not cross the memory index

    return idxs

  def _retrieve_batch(self, idxs, n, L):
    vec_idxs = idxs.transpose().reshape(-1)  # Unroll indices
    observations = torch.as_tensor(self.observations[vec_idxs].astype(np.float32))
    # if not self.symbolic_env:
    #   preprocess_observation_(observations, self.bit_depth)  # Undo discretisation for visual observations
    # return observations.reshape(L, n, *observations.shape[1:]), self.actions[vec_idxs].reshape(L, n, -1), self.rewards[vec_idxs].reshape(L, n), self.nonterminals[vec_idxs].reshape(L, n, 1)
    # TODO: remove the 1 dim of nonterminal
    return observations.reshape(L, n, *observations.shape[1:]), self.actions[vec_idxs].reshape(L, n, -1), self.rewards[vec_idxs].reshape(L, n), self.nonterminals[vec_idxs].reshape(L, n)
  # Returns a batch of sequence chunks uniformly sampled from the memory

  def sample(self, n, L):
    _sample_idx = []
    for _ in range(n):
      _i = np.random.randint(n)
      if _i < n//4:  # force to sample with ends state
        _sample_idx.append(self._sample_idx(L, need_ends=True))

        # print("check sampled data with ends", self.rewards[self._sample_idx(L, need_ends=True)])
      else:  # random sample
        _sample_idx.append(self._sample_idx(L))

    batch = self._retrieve_batch(np.asarray(_sample_idx), n, L)
    # batch = self._retrieve_batch(np.asarray([self._sample_idx(L) for _ in range(n)]), n, L)
    return [torch.as_tensor(item).to(device=self.device) for item in batch]
