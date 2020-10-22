import argparse
import os
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from env import CONTROL_SUITE_ENVS, Env, GYM_ENVS, DONKEY_CAR_ENVS, EnvBatcher
from agent import Dreamer, bottle
from utils import lineplot, write_video
from tensorboardX import SummaryWriter
import wandb

# Hyperparameters
parser = argparse.ArgumentParser(description='Dreamer')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--env', type=str, default='donkey-generated-roads-v0',
										choices=GYM_ENVS + CONTROL_SUITE_ENVS + DONKEY_CAR_ENVS,
										help='Gym/Control Suite/Donkey_Car environment')
parser.add_argument('--symbolic', action='store_true', help='Symbolic features')
parser.add_argument('--max-episode-length', type=int, default=1000, metavar='T', help='Max episode length')
parser.add_argument('--experience-size', type=int, default=1000000, metavar='D',
										help='Experience replay size')  # Original implementation has an unlimited buffer size, but 1 million is the max experience collected anyway
parser.add_argument('--cnn-act', type=str, default='relu', choices=dir(F),
										help='Model activation function for a convolution layer')
parser.add_argument('--dense-act', type=str, default='elu', choices=dir(F),
										help='Model activation function a dense layer')
parser.add_argument('--embedding-size', type=int, default=512, metavar='E',
										help='Observation embedding size')  # Note that the default encoder for visual observations outputs a 1024D vector; for other embedding sizes an additional fully-connected layer is used
parser.add_argument('--hidden-size', type=int, default=300, metavar='H',
										help='Hidden size')  # paper:300; tf_implementation:400; aligned wit paper.
parser.add_argument('--belief-size', type=int, default=200, metavar='H', help='Belief/hidden size')
parser.add_argument('--state-size', type=int, default=30, metavar='Z', help='State/latent size')
parser.add_argument('--action-repeat', type=int, default=1, metavar='R', help='Action repeat')
parser.add_argument('--episodes', type=int, default=50, metavar='E', help='Total number of episodes')

parser.add_argument('--seed-episodes', type=int, default=5, metavar='S', help='Seed episodes')

parser.add_argument('--collect-interval', type=int, default=100, metavar='C', help='Collect interval')

parser.add_argument('--batch-size', type=int, default=50, metavar='B', help='Batch size')
parser.add_argument('--chunk-size', type=int, default=50, metavar='L', help='Chunk size')
parser.add_argument('--free-nats', type=float, default=3, metavar='F', help='Free nats')
parser.add_argument('--bit-depth', type=int, default=8, metavar='B', help='Image bit depth (quantisation)')
parser.add_argument('--world_lr', type=float, default=6e-4, metavar='α', help='Learning rate')
parser.add_argument('--actor_lr', type=float, default=8e-5, metavar='α', help='Learning rate')
parser.add_argument('--value_lr', type=float, default=8e-5, metavar='α', help='Learning rate')
# parser.add_argument('--world_lr', type=float, default=1e-3, metavar='α', help='Learning rate')
# parser.add_argument('--actor_lr', type=float, default=1e-4, metavar='α', help='Learning rate')
# parser.add_argument('--value_lr', type=float, default=1e-4, metavar='α', help='Learning rate')
parser.add_argument('--learning-rate-schedule', type=int, default=0, metavar='αS',
										help='Linear learning rate schedule (optimisation steps from 0 to final learning rate; 0 to disable)')
parser.add_argument('--adam-epsilon', type=float, default=1e-7, metavar='ε', help='Adam optimizer epsilon value')
# Note that original has a linear learning rate decay, but it seems unlikely that this makes a significant difference
parser.add_argument('--grad-clip-norm', type=float, default=100.0, metavar='C', help='Gradient clipping norm')
parser.add_argument('--expl_amount', type=float, default=0.3, help='exploration noise')

parser.add_argument('--planning-horizon', type=int, default=15, metavar='H', help='Planning horizon distance')

parser.add_argument('--discount', type=float, default=0.99, metavar='H', help='Planning horizon distance')
parser.add_argument('--disclam', type=float, default=0.95, metavar='H', help='discount rate to compute return')

parser.add_argument('--test', action='store_true', help='Test only')
parser.add_argument('--test-interval', type=int, default=5, metavar='I', help='Test interval (episodes)')
parser.add_argument('--test-episodes', type=int, default=1, metavar='E', help='Number of test episodes')  # for donkey env, since it can only use one car, just use test_epi=1 here, in other env, 10
parser.add_argument('--checkpoint-interval', type=int, default=500, metavar='I', help='Checkpoint interval (episodes)')
parser.add_argument('--checkpoint-experience', action='store_true', help='Checkpoint experience replay')
parser.add_argument('--models', type=str, default='', metavar='M', help='Load model checkpoint')
parser.add_argument('--experience-replay', type=str, default='', metavar='ER', help='Load experience replay')
parser.add_argument('--render', action='store_true', help='Render environment')
# For pcont
parser.add_argument('--pcont', action='store_true',
										help='Wheter to predict the continuity, used to handle the terminal state')
parser.add_argument('--pcont_scale', type=int, default=10, help='The coefficient term of the pcont loss')
parser.add_argument('--reward_scale', type=int, default=10, help='the coefficient term of reward loss')
# For donkey car
parser.add_argument('--sim_path', type=str,
										default='/u/95/zhaoy13/unix/summer/ICRA/donkey/DonkeySimLinux/donkey_sim.x86_64',
										help='path to the unity simulator, a .x86_64 file.')
parser.add_argument('--port', type=int, default=9091, help='port to use for tcp')
parser.add_argument('--host', type=str, default='127.0.0.1', help='host ip')
# por sac
parser.add_argument('--with_logprob', action='store_true')
parser.add_argument('--use_automatic_entropy_tuning', action='store_true', help="Use the entropy regularization")
parser.add_argument('--temp', type=float, default=0.003)  # temp for entropy

parser.add_argument('--action_size', default=2)
parser.add_argument('--observation_size', default=(1, 40, 40))

# for action constrains
parser.add_argument('--fix_speed', action='store_true')
parser.add_argument('--throttle_base', default=0.3)
parser.add_argument('--throttle_min', default=0.1)
parser.add_argument('--throttle_max', default=0.5)
parser.add_argument('--angle_min', default=-1)
parser.add_argument('--angle_max', default=1)
args = parser.parse_args()

wandb.init(project="donkey_sac")
wandb.config.update(args)

print(' ' * 26 + 'Options')
for k, v in vars(args).items():
	print(' ' * 26 + k + ': ' + str(v))

# Setup
results_dir = os.path.join('results', args.env, str(args.seed))
os.makedirs(results_dir, exist_ok=True)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available() and not args.disable_cuda:
	args.device = torch.device('cuda')
	torch.cuda.manual_seed(args.seed)
else:
	args.device = torch.device('cpu')

metrics = {'steps': [], 'episodes': [], 'train_rewards': [], 'test_episodes': [], 'test_rewards': [],
					 'observation_loss': [], 'reward_loss': [], 'kl_loss': [], 'actor_loss': [], 'value_loss': []}

summary_name = results_dir + "/{}_{}_log"
writer = SummaryWriter(summary_name.format(args.env, args.seed))

# Initialise training environment and experience replay memory
env = Env(args.env, args.symbolic, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth,
					sim_path=args.sim_path, host=args.host, port=args.port)
agent = Dreamer(args)

if args.experience_replay is not '' and os.path.exists(args.experience_replay):
	agent.D = torch.load(args.experience_replay)
	metrics['steps'], metrics['episodes'] = [agent.D.steps] * agent.D.episodes, list(range(1, agent.D.episodes + 1))
elif not args.test:
	# agent.D = ExperienceReplay(args.experience_size, args.symbolic, env.observation_size, env.action_size, args.bit_depth,
	# 										 args.device)  # TODO: add env in the argument list

	# Initialise dataset D with S random seed episodes
	for s in range(1, args.seed_episodes + 1):
		observation, done, t = env.reset(), False, 0
		while not done:
			action = env.sample_random_action()
			action[1] = 0.3  # fix the action
			next_observation, reward, done = env.step(action)
			# agent.D.append(observation, action, reward, done)
			agent.D.append(next_observation, action, reward, done)  # TODO:1
			observation = next_observation
			t += 1
		metrics['steps'].append(t * args.action_repeat + (0 if len(metrics['steps']) == 0 else metrics['steps'][-1]))
		metrics['episodes'].append(s)



# if args.models is not '' and os.path.exists(args.models):
# 	agent.load_models()
# 	model_dicts = torch.load(args.models)
# 	transition_model.load_state_dict(model_dicts['transition_model'])
# 	observation_model.load_state_dict(model_dicts['observation_model'])
# 	reward_model1.load_state_dict(model_dicts['reward_model1'])
# 	encoder.load_state_dict(model_dicts['encoder'])
# 	actor_model.load_state_dict(model_dicts['actor_model'])
# 	value_model1.load_state_dict(model_dicts['value_model1'])
# 	value_model2.load_state_dict(model_dicts['value_model2'])
# 	world_optimizer.load_state_dict(model_dicts['world_optimizer'])


# def update_belief_and_act(args, env, actor_model, transition_model, encoder, belief, posterior_state, action,
# 													observation, deterministic=False):
# 	# Infer belief over current state q(s_t|o≤t,a<t) from the history
# 	belief, _, _, _, posterior_state, _, _ = transition_model(
# 		posterior_state,
# 		action.unsqueeze(dim=0),
# 		belief,
# 		encoder(observation).unsqueeze(dim=0))  # Action and observation need extra time dimension
#
# 	belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(
# 		dim=0)  # Remove time dimension from belief/state
#
# 	#
# 	# if explore:
# 	#   action = actor_model(belief, posterior_state).rsample()  # batch_shape=1, event_shape=6
# 	#   # add exploration noise -- following the original code: line 275-280
# 	#   action = Normal(action, args.expl_amount).rsample()
# 	#
# 	#   # TODO: add this later
# 	#   # action = torch.clamp(action, [-1.0, 0.0], [1.0, 5.0])
# 	# else:
# 	#   action = actor_model(belief, posterior_state).mode()
# 	action = actor_model(belief, posterior_state, deterministic=deterministic,
# 											 with_logprob=False)  # with sac, not need to add exploration noise, the max entropy can maintain it.
# 	if args.temp == 0 and not deterministic:
# 		action = Normal(action, args.expl_amount).rsample()
# 	action[:, 1] = 0.3  # TODO: fix the speed
# 	next_observation, reward, done = env.step(action.cpu() if isinstance(env, EnvBatcher) else action[
# 		0].cpu())  # Perform environment step (action repeats handled internally)
#
# 	print(bottle(value_model1, (belief.unsqueeze(dim=0), posterior_state.unsqueeze(dim=0))).item())
# 	return belief, posterior_state, action, next_observation, reward, done

#
# # Testing only
# if args.test:
# 	# Set models to eval mode
# 	transition_model.eval()
# 	reward_model1.eval()
# 	reward_model2.eval()
# 	encoder.eval()
# 	with torch.no_grad():
# 		total_reward = 0
# 		for _ in tqdm(range(args.test_episodes)):
# 			observation = env.reset()
#
# 			belief = torch.zeros(1, args.belief_size, device=args.device)
# 			posterior_state = torch.zeros(1, args.state_size, device=args.device)
# 			action = torch.zeros(1, env.action_size, device=args.device)
#
# 			pbar = tqdm(range(args.max_episode_length // args.action_repeat))
# 			for t in pbar:
# 				belief, posterior_state, action, observation, reward, done = update_belief_and_act(
# 					args,
# 					env,
# 					actor_model,
# 					transition_model,
# 					encoder, belief,
# 					posterior_state, action,
# 					observation.to(device=args.device),
# 					deterministic=True)
#
# 				total_reward += reward
#
# 				if args.render:
# 					env.render()
# 				if done:
# 					pbar.close()
# 					break
#
# 	print('Average Reward:', total_reward / args.test_episodes)
# 	env.close()
# 	quit()

# Training (and testing)
for episode in tqdm(range(metrics['episodes'][-1] + 1, args.episodes + 1), total=args.episodes,
										initial=metrics['episodes'][-1] + 1):
	# Model fitting
	loss_info = agent.update_parameters(args.collect_interval)

	# Update and plot loss metrics
	losses = tuple(zip(*loss_info))
	metrics['observation_loss'].append(losses[0])
	metrics['reward_loss'].append(losses[1])
	metrics['kl_loss'].append(losses[2])
	metrics['kl_loss'].append(losses[3])
	metrics['actor_loss'].append(losses[4])
	metrics['value_loss'].append(losses[5])
	lineplot(metrics['episodes'][-len(metrics['observation_loss']):], metrics['observation_loss'], 'observation_loss',
					 results_dir)
	lineplot(metrics['episodes'][-len(metrics['reward_loss']):], metrics['reward_loss'], 'reward_loss', results_dir)
	lineplot(metrics['episodes'][-len(metrics['kl_loss']):], metrics['kl_loss'], 'kl_loss', results_dir)
	lineplot(metrics['episodes'][-len(metrics['actor_loss']):], metrics['actor_loss'], 'actor_loss', results_dir)
	lineplot(metrics['episodes'][-len(metrics['value_loss']):], metrics['value_loss'], 'value_loss', results_dir)

	# Data collection
	with torch.no_grad():
		observation, total_reward = env.reset(), 0
		belief = torch.zeros(1, args.belief_size, device=args.device)
		posterior_state = torch.zeros(1, args.state_size, device=args.device)
		action = torch.zeros(1, env.action_size, device=args.device)

		pbar = tqdm(range(args.max_episode_length // args.action_repeat))
		for t in pbar:
			# maintain belief and posterior_state
			belief, posterior_state = agent.infer_state(observation.to(device=args.device), action, belief, posterior_state)
			action = agent.select_action((belief, posterior_state), deterministic=False)

			# interact with env
			next_observation, reward, done = env.step(action.cpu() if isinstance(env, EnvBatcher) else action[
					0].cpu())  # Perform environment step (action repeats handled internally)

			# agent.D.append(observation, action.cpu(), reward, done)
			agent.D.append(next_observation, action.cpu(), reward, done)  # TODO:2
			total_reward += reward
			observation = next_observation
			print(bottle(agent.value_model, (belief.unsqueeze(dim=0), posterior_state.unsqueeze(dim=0))).item())
			if args.render:
				env.render()
			if done:
				pbar.close()
				break

		# Update and plot train reward metrics
		metrics['steps'].append(t + metrics['steps'][-1])
		metrics['episodes'].append(episode)
		metrics['train_rewards'].append(total_reward)
		lineplot(metrics['episodes'][-len(metrics['train_rewards']):], metrics['train_rewards'], 'train_rewards',
						 results_dir)

	# Test model
	if episode % args.test_interval == 0:
		# Set models to eval mode
		agent.transition_model.eval()
		agent.observation_model.eval()
		agent.reward_model.eval()
		agent.encoder.eval()
		agent.actor_model.eval()
		agent.value_model.eval()

		# Initialise parallelised test environments
		# test_envs = EnvBatcher(
		#   Env,
		#   (args.env, args.symbolic, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth, args.sim_path, args.host, args.port),
		#   {},
		#   args.test_episodes)

		with torch.no_grad():
			# observation = test_envs.reset()
			observation = env.reset()
			# total_rewards = np.zeros((args.test_episodes, ))
			total_rewards = 0
			video_frames = []

			belief = torch.zeros(args.test_episodes, args.belief_size, device=args.device)
			posterior_state = torch.zeros(args.test_episodes, args.state_size, device=args.device)
			action = torch.zeros(args.test_episodes, env.action_size, device=args.device)

			for t in tqdm(range(args.max_episode_length // args.action_repeat)):
				belief, posterior_state = agent.infer_state(observation.to(device=args.device), action, belief, posterior_state)
				action = agent.select_action((belief, posterior_state), deterministic=True)

				# interact with env
				next_observation, reward, done = env.step(action.cpu() if isinstance(env, EnvBatcher) else action[
					0].cpu())  # Perform environment step (action repeats handled internally)

				# total_rewards += reward.numpy()
				total_rewards += reward
				if not args.symbolic:  # Collect real vs. predicted frames for video
					video_frames.append(
						make_grid(torch.cat([observation, agent.observation_model(belief, posterior_state).cpu()], dim=3) + 0.5,
											nrow=5).numpy())  # Decentre
				observation = next_observation
				# if done.sum().item() == args.test_episodes:
				#   pbar.close()
				#   break
				if done:
					pbar.close()
					break

		# Update and plot reward metrics (and write video if applicable) and save metrics
		metrics['test_episodes'].append(episode)
		# metrics['test_rewards'].append(total_rewards.tolist())
		metrics['test_rewards'].append(total_rewards)
		lineplot(metrics['test_episodes'], metrics['test_rewards'], 'test_rewards', results_dir)
		lineplot(np.asarray(metrics['steps'])[np.asarray(metrics['test_episodes']) - 1], metrics['test_rewards'],
						 'test_rewards_steps', results_dir, xaxis='step')
		if not args.symbolic:
			episode_str = str(episode).zfill(len(str(args.episodes)))
			write_video(video_frames, 'test_episode_%s' % episode_str, results_dir)  # Lossy compression
			save_image(torch.as_tensor(video_frames[-1]), os.path.join(results_dir, 'test_episode_%s.png' % episode_str))
		torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))

		# Set models to train mode
		agent.transition_model.train()
		agent.observation_model.train()
		agent.reward_model.train()
		agent.encoder.train()
		agent.actor_model.train()
		agent.value_model.train()
	# Close test environments
	# test_envs.close()
	# env.close()

	writer.add_scalar("train_reward", metrics['train_rewards'][-1], metrics['steps'][-1])
	writer.add_scalar("train/episode_reward", metrics['train_rewards'][-1], metrics['steps'][-1] * args.action_repeat)
	writer.add_scalar("observation_loss", metrics['observation_loss'][0][-1], metrics['steps'][-1])
	writer.add_scalar("reward_loss", metrics['reward_loss'][0][-1], metrics['steps'][-1])
	writer.add_scalar("kl_loss", metrics['kl_loss'][0][-1], metrics['steps'][-1])
	writer.add_scalar("actor_loss", metrics['actor_loss'][0][-1], metrics['steps'][-1])
	writer.add_scalar("value_loss", metrics['value_loss'][0][-1], metrics['steps'][-1])
	print("episodes: {}, total_steps: {}, train_reward: {} ".format(metrics['episodes'][-1], metrics['steps'][-1],
																																	metrics['train_rewards'][-1]))
	wandb.log({"episode": episode, "cumulative_reward": total_reward})
	# Checkpoint models
	if episode % args.checkpoint_interval == 0:
		torch.save({'transition_model': agent.transition_model.state_dict(),
								'observation_model': agent.observation_model.state_dict(),
								'reward_model1': agent.reward_model.state_dict(),
								'encoder': agent.encoder.state_dict(),
								'actor_model': agent.actor_model.state_dict(),
								'value_model1': agent.value_model.state_dict(),
								'value_model2': agent.value_model.state_dict(),
								'world_optimizer': agent.world_optimizer.state_dict(),
								'actor_optimizer': agent.actor_optimizer.state_dict(),
								'value_optimizer': agent.value_optimizer.state_dict()
								}, os.path.join(results_dir, 'models_%d.pth' % episode))
		if args.checkpoint_experience:
			torch.save(agent.D, os.path.join(results_dir,
																 'experience.pth'))  # Warning: will fail with MemoryError with large memory sizes

# Close training environment
env.close()
