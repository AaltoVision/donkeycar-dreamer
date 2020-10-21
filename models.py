from typing import Optional, List
import torch
from torch import jit, nn
from torch.nn import functional as F
import torch.distributions
from torch.distributions.normal import Normal
from torch.distributions.transforms import SigmoidTransform, AffineTransform
from torch.distributions.transformed_distribution import TransformedDistribution
import numpy as np


# Wraps the input tuple for a function to process a time x batch x features sequence in batch x features (assumes one output)
def bottle(f, x_tuple):
  x_sizes = tuple(map(lambda x: x.size(), x_tuple))
  y = f(*map(lambda x: x[0].view(x[1][0] * x[1][1], *x[1][2:]), zip(x_tuple, x_sizes)))
  y_size = y.size()
  output = y.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:])
  return output


class TransitionModel(nn.Module):
  __constants__ = ['min_std_dev']

  def __init__(self, belief_size, state_size, action_size, hidden_size, embedding_size, activation_function='relu', min_std_dev=0.1):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.min_std_dev = min_std_dev
    self.fc_embed_state_action = nn.Linear(state_size + action_size, belief_size)
    self.rnn = nn.GRUCell(belief_size, belief_size)
    self.fc_embed_belief_prior = nn.Linear(belief_size, hidden_size)
    self.fc_state_prior = nn.Linear(hidden_size, 2 * state_size)
    self.fc_embed_belief_posterior = nn.Linear(belief_size + embedding_size, hidden_size)
    self.fc_state_posterior = nn.Linear(hidden_size, 2 * state_size)
    self.modules = [self.fc_embed_state_action, self.fc_embed_belief_prior, self.fc_state_prior, self.fc_embed_belief_posterior, self.fc_state_posterior]

  # Operates over (previous) state, (previous) actions, (previous) belief, (previous) nonterminals (mask), and (current) observations
  # Diagram of expected inputs and outputs for T = 5 (-x- signifying beginning of output belief/state that gets sliced off):
  # t :  0  1  2  3  4  5
  # o :    -X--X--X--X--X-
  # a : -X--X--X--X--X-
  # n : -X--X--X--X--X-
  # pb: -X-
  # ps: -X-
  # b : -x--X--X--X--X--X-
  # s : -x--X--X--X--X--X-
  # @jit.script_method
  def forward(self, prev_state:torch.Tensor, actions:torch.Tensor, prev_belief:torch.Tensor, observations:Optional[torch.Tensor]=None, nonterminals:Optional[torch.Tensor]=None) -> List[torch.Tensor]:
    '''
    Input: init_belief, init_state:  torch.Size([50, 200]) torch.Size([50, 30])
    Output: beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs
            torch.Size([49, 50, 200]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30])
    '''
    # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
    T = actions.size(0) + 1
    beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T
    beliefs[0], prior_states[0], posterior_states[0] = prev_belief, prev_state, prev_state
    # Loop over time sequence
    for t in range(T - 1):
      _state = prior_states[t] if observations is None else posterior_states[t]  # Select appropriate previous state
      _state = _state if (nonterminals is None or t == 0) else _state * nonterminals[t-1].unsqueeze(dim=-1)  # Mask if previous transition was terminal  # TODO:3
      # Compute belief (deterministic hidden state)
      hidden = self.act_fn(self.fc_embed_state_action(torch.cat([_state, actions[t]], dim=1)))
      beliefs[t + 1] = self.rnn(hidden, beliefs[t])
      # Compute state prior by applying transition dynamics
      hidden = self.act_fn(self.fc_embed_belief_prior(beliefs[t + 1]))
      prior_means[t + 1], _prior_std_dev = torch.chunk(self.fc_state_prior(hidden), 2, dim=1)
      prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + self.min_std_dev
      prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])     
      if observations is not None:
        # Compute state posterior by applying transition dynamics and using current observation
        t_ = t - 1  # Use t_ to deal with different time indexing for observations
        hidden = self.act_fn(self.fc_embed_belief_posterior(torch.cat([beliefs[t + 1], observations[t_ + 1]], dim=1)))
        posterior_means[t + 1], _posterior_std_dev = torch.chunk(self.fc_state_posterior(hidden), 2, dim=1)
        posterior_std_devs[t + 1] = F.softplus(_posterior_std_dev) + self.min_std_dev
        posterior_states[t + 1] = posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.randn_like(posterior_means[t + 1])
    # Return new hidden states
    hidden = [torch.stack(beliefs[1:], dim=0), torch.stack(prior_states[1:], dim=0), torch.stack(prior_means[1:], dim=0), torch.stack(prior_std_devs[1:], dim=0)]
    if observations is not None:
      hidden += [torch.stack(posterior_states[1:], dim=0), torch.stack(posterior_means[1:], dim=0), torch.stack(posterior_std_devs[1:], dim=0)]
    return hidden


class SymbolicObservationModel(jit.ScriptModule):
  def __init__(self, observation_size, belief_size, state_size, embedding_size, activation_function='relu'):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
    self.fc2 = nn.Linear(embedding_size, embedding_size)
    self.fc3 = nn.Linear(embedding_size, observation_size)

  @jit.script_method
  def forward(self, belief, state):
    hidden = self.act_fn(self.fc1(torch.cat([belief, state], dim=1)))
    hidden = self.act_fn(self.fc2(hidden))
    observation = self.fc3(hidden)
    return observation


class VisualObservationModel(jit.ScriptModule):
  __constants__ = ['embedding_size']
  
  def __init__(self, belief_size, state_size, embedding_size, activation_function='relu'):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.embedding_size = embedding_size
    self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
    self.conv1 = nn.ConvTranspose2d(embedding_size, 128, 5, stride=2)
    self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
    self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
    self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)

  @jit.script_method
  def forward(self, belief, state):
    hidden = self.fc1(torch.cat([belief, state], dim=1))  # No nonlinearity here
    hidden = hidden.view(-1, self.embedding_size, 1, 1)
    hidden = self.act_fn(self.conv1(hidden))
    hidden = self.act_fn(self.conv2(hidden))
    hidden = self.act_fn(self.conv3(hidden))
    observation = self.conv4(hidden)
    return observation


def ObservationModel(symbolic, observation_size, belief_size, state_size, embedding_size, activation_function='relu'):
  if symbolic:
    return SymbolicObservationModel(observation_size, belief_size, state_size, embedding_size, activation_function)
  else:
    return VisualObservationModel(belief_size, state_size, embedding_size, activation_function)


class RewardModel(jit.ScriptModule):
  def __init__(self, belief_size, state_size, hidden_size, activation_function='relu'):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, 1)

  @jit.script_method
  def forward(self, belief, state):
    x = torch.cat([belief, state],dim=1)
    hidden = self.act_fn(self.fc1(x))
    hidden = self.act_fn(self.fc2(hidden))
    reward = self.fc3(hidden)
    reward = reward.squeeze(dim=-1)
    return reward


class ValueModel(nn.Module):
  def __init__(self, belief_size, state_size, hidden_size, activation_function='relu'):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, hidden_size)
    self.fc4 = nn.Linear(hidden_size, 1)

  def forward(self, belief, state):
    x = torch.cat([belief, state],dim=1)
    hidden = self.act_fn(self.fc1(x))
    hidden = self.act_fn(self.fc2(hidden))
    hidden = self.act_fn(self.fc3(hidden))
    reward = self.fc4(hidden).squeeze(dim=1)
    return reward


class SymbolicEncoder(jit.ScriptModule):
  def __init__(self, observation_size, embedding_size, activation_function='relu'):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.fc1 = nn.Linear(observation_size, embedding_size)
    self.fc2 = nn.Linear(embedding_size, embedding_size)
    self.fc3 = nn.Linear(embedding_size, embedding_size)

  @jit.script_method
  def forward(self, observation):
    hidden = self.act_fn(self.fc1(observation))
    hidden = self.act_fn(self.fc2(hidden))
    hidden = self.fc3(hidden)
    return hidden


class VisualEncoder(jit.ScriptModule):
  __constants__ = ['embedding_size']
  
  def __init__(self, embedding_size, activation_function='relu'):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.embedding_size = embedding_size
    self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
    self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
    self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
    self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
    self.fc = nn.Identity() if embedding_size == 1024 else nn.Linear(1024, embedding_size)

  @jit.script_method
  def forward(self, observation):
    hidden = self.act_fn(self.conv1(observation))
    hidden = self.act_fn(self.conv2(hidden))
    hidden = self.act_fn(self.conv3(hidden))
    hidden = self.act_fn(self.conv4(hidden))
    hidden = hidden.view(-1, 1024)
    hidden = self.fc(hidden)  # Identity if embedding size is 1024 else linear projection
    return hidden


def Encoder(symbolic, observation_size, embedding_size, activation_function='relu'):
  if symbolic:
    return SymbolicEncoder(observation_size, embedding_size, activation_function)
  else:
    return VisualEncoder(embedding_size, activation_function)


class ActorModel(nn.Module):
  def __init__(self, action_size, belief_size, state_size, hidden_size, mean_scale=5, min_std=1e-4, init_std=5, activation_function="elu",
               fix_speed=False, throttle_base=0.3):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, hidden_size)
    self.fc4 = nn.Linear(hidden_size, hidden_size)

    # self.fc5 = nn.Linear(hidden_size, 2 * (action_size-1)) # TODO: only predict the direction, remove this latter
    self.fc5 = nn.Linear(hidden_size, 2 * (action_size))
    self.min_std = min_std
    self.init_std = init_std
    self.mean_scale = mean_scale

    self.fix_speed = fix_speed
    self.throtlle_base = throttle_base
  # def forward(self, belief, state, deterministic=False, with_logprob=False):
  #   hidden = self.act_fn(self.fc1(torch.cat([belief, state], dim=-1)))
  #   # print("first hidden", hidden[0][1], self.fc1.weight[1])
  #   hidden = self.act_fn(self.fc2(hidden))
  #   hidden = self.act_fn(self.fc3(hidden))
  #   hidden = self.act_fn(self.fc4(hidden))
  #   # print("middle hidden", self.fc4.weight)
  #   hidden = self.fc5(hidden)
  #
  #   raw_init_std = np.log(np.exp(self.init_std) - 1)
  #   mean, std = torch.chunk(hidden, 2, dim=-1)
  #   mean = self.mean_scale * torch.tanh(
  #     mean / self.mean_scale)  # bound the action to [-5, 5] --> to avoid numerical instabilities.  For computing log-probabilities, we need to invert the tanh and this becomes difficult in highly saturated regions.
  #   std = F.softplus(std + raw_init_std) + self.min_std
  #   pi_dist = torch.distributions.Normal(mean, std)
  #
  #   # mean, log_std = torch.chunk(hidden, 2, dim=-1)
  #   # log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
  #   # std = torch.exp(log_std)
  #
  #   # pi_dist = torch.distributions.Normal(mean, std)
  #
  #   if deterministic:
  #     pi_action = mean
  #   else:
  #     pi_action = pi_dist.rsample()  # shape [batch*(chunk-1), act_dim]
  #
  #   if with_logprob:
  #     logp_pi = pi_dist.log_prob(pi_action).sum(axis=-1)  # shape [batch*(chunk-1)]
  #     # print("check action.shape", pi_action.shape)
  #     # print("logp.shape", logp_pi.shape)
  #     # print("before sum(axis=1)", (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).shape)
  #     logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
  #   else:
  #     logp_pi = None
  #
  #   pi_action = torch.tanh(pi_action)
  #
  #   return pi_action, logp_pi


  def forward(self, belief, state, deterministic=False, with_logprob=False,):
    raw_init_std = np.log(np.exp(self.init_std) - 1)
    hidden = self.act_fn(self.fc1(torch.cat([belief, state], dim=-1)))
    hidden = self.act_fn(self.fc2(hidden))
    hidden = self.act_fn(self.fc3(hidden))
    hidden = self.act_fn(self.fc4(hidden))
    hidden = self.fc5(hidden)
    mean, std = torch.chunk(hidden, 2, dim=-1)

    # # ---------
    # mean = self.mean_scale * torch.tanh(mean / self.mean_scale)  # bound the action to [-5, 5] --> to avoid numerical instabilities.  For computing log-probabilities, we need to invert the tanh and this becomes difficult in highly saturated regions.
    # speed = torch.full(mean.shape, 0.3).to("cuda")
    # mean = torch.cat((mean, speed), -1)
    #
    # std = F.softplus(std + raw_init_std) + self.min_std
    #
    # speed = torch.full(std.shape, 0.0).to("cuda")
    # std = torch.cat((std, speed), -1)
    #
    # dist = torch.distributions.Normal(mean, std)
    # transform = [torch.distributions.transforms.TanhTransform()]
    # dist = torch.distributions.TransformedDistribution(dist, transform)
    # dist = torch.distributions.independent.Independent(dist, 1)  # Introduces dependence between actions dimension
    # dist = SampleDist(dist)  # because after transform a distribution, some methods may become invalid, such as entropy, mean and mode, we need SmapleDist to approximate it.
    # return dist  # dist ~ tanh(Normal(mean, std)); remember when sampling, using rsample() to adopt the reparameterization trick


    mean = self.mean_scale * torch.tanh(mean / self.mean_scale)  # bound the action to [-5, 5] --> to avoid numerical instabilities.  For computing log-probabilities, we need to invert the tanh and this becomes difficult in highly saturated regions.
    std = F.softplus(std + raw_init_std) + self.min_std

    dist = torch.distributions.Normal(mean, std)
    # TanhTransform = ComposeTransform([AffineTransform(0., 2.), SigmoidTransform(), AffineTransform(-1., 2.)])
    if self.fix_speed:
      transform = [AffineTransform(0., 2.), SigmoidTransform(), AffineTransform(-1., 2.),  # TanhTransform
                   AffineTransform(loc=torch.tensor([0.0, self.throtlle_base]).to("cuda"),
                                  scale=torch.tensor([1.0, 0.0]).to("cuda"))]  # TODO: this is limited at donkeycar env
    else:
      transform = [AffineTransform(0., 2.), SigmoidTransform(), AffineTransform(-1., 2.),  # TanhTransform
                   AffineTransform(loc=torch.tensor([0.0, self.throtlle_base]).to("cuda"),
                                  scale=torch.tensor([1.0, 0.2]).to("cuda"))]  # TODO: this is limited at donkeycar env

    dist = TransformedDistribution(dist, transform)
    # dist = torch.distributions.independent.Independent(dist, 1)  # Introduces dependence between actions dimension
    dist = SampleDist(dist)  # because after transform a distribution, some methods may become invalid, such as entropy, mean and mode, we need SmapleDist to approximate it.

    if deterministic:
      action = dist.mean
    else:
      action = dist.rsample()

    logp_pi = dist.log_prob(action)
    print(logp_pi[:, 0])
    # not use logprob now
    if with_logprob:
      if self.fix_speed:  # if fix speed, the entropy of speed is nan # TODO: problem here
        logp_pi = dist.log_prob(action)[:, 0]  # only consider the entropy of the
      else:
        logp_pi = dist.log_prob(action).sum(dim=1)
    else:
      logp_pi = None
    # action dim: [batch, act_dim], log_pi dim:[batch]
    return action, logp_pi  # dist ~ tanh(Normal(mean, std)); remember when sampling, using rsample() to adopt the reparameterization trick



class SampleDist:
  """
  After TransformedDistribution, many methods becomes invalid, therefore, we need to approximate them.
  """
  def __init__(self, dist: torch.distributions.Distribution, samples=100):
    self._dist = dist
    self._samples = samples

  @property
  def name(self):
    return 'SampleDist'

  def __getattr__(self, name):
    return getattr(self._dist, name)

  @property
  def mean(self):
    dist = self._dist.expand((self._samples, *self._dist.batch_shape))
    sample = dist.rsample()
    return torch.mean(sample, 0)

  def mode(self):
    dist = self._dist.expand((self._samples, *self._dist.batch_shape))
    sample = dist.rsample()
    # print("dist in mode", sample.shape)
    logprob = dist.log_prob(sample)
    batch_size = sample.size(1)
    feature_size = sample.size(2)
    indices = torch.argmax(logprob, dim=0).reshape(1, batch_size, 1).expand(1, batch_size, feature_size)
    return torch.gather(sample, 0, indices).squeeze(0)

  def entropy(self):
    dist = self._dist.expand((self._samples, *self._dist.batch_shape))
    sample = dist.rsample()
    logprob = dist.log_prob(sample)
    return -torch.mean(logprob, 0)


class PCONTModel(nn.Module):
  """ predict the prob of whether a state is a terminal state. """
  def __init__(self, belief_size, state_size, hidden_size, activation_function='relu'):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, hidden_size)
    self.fc4 = nn.Linear(hidden_size, 1)


  def forward(self, belief, state):
    x = torch.cat([belief, state],dim=1)
    hidden = self.act_fn(self.fc1(x))
    hidden = self.act_fn(self.fc2(hidden))
    hidden = self.act_fn(self.fc3(hidden))
    x = self.fc4(hidden).squeeze(dim=1)
    p = torch.sigmoid(x)
    return p
