import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib
import matplotlib.pyplot as plt



gamma = 0.99
seed = 543
log_interval = 10

parser = argparse.ArgumentParser(description='args')

parser.add_argument('--render', action='store_true',
					help='render the environment')
args = parser.parse_args()

env = gym.make('CartPole-v1')
env.seed(seed)
torch.manual_seed(seed)

x_data = []
y_data = []

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
	from IPython import display
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


def plot_figure(fig_id = 2, fig_name = 'training', x_label = 'episodes', y_label = 'rewards', x = x_data, y = y_data):

	plt.figure(fig_id)
	plt.clf()

	plt.title(fig_name)
	plt.xlabel(x_label)
	plt.ylabel(y_label)

	plt.plot(x,y)

	plt.pause(0.001)

	if is_ipython:
		display.clear_output(wait=True)
		display.display(plt.gcf())



class Policy(nn.Module):

	def __init__(self):
		super(Policy, self).__init__()

		self.layer1 = nn.Linear(4, 128)

		self.branch1 = nn.Linear(128, 2)

		self.branch2 = nn.Linear(128, 1)

		self.saved_actions = []

		self.rewards = []

	def forward(self, x):

		x = F.relu(self.layer1(x))

		action_prob = F.softmax(self.branch1(x), dim = -1)

		state_values = self.branch2(x)

		return action_prob, state_values


model = Policy()
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()

def select_action(state):

	state = torch.from_numpy(state).float()
	probs, state_value = model(state)

	m = Categorical(probs)

	action = m.sample()

	model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

	return action.item()

def finish_episode():

	R = 0
	saved_actions = model.saved_actions
	policy_losses = []
	value_losses = []
	returns = []

	for r in model.rewards[::-1]:
		R = r + gamma * R
		returns.insert(0, R)

	returns = torch.tensor(returns)
	returns = (returns - returns.mean()) / (returns.std() + eps)

	for (log_prob, value), R in zip(saved_actions, returns):
		advantage = R - value.item()

		policy_losses.append(-log_prob * advantage)

		value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

	optimizer.zero_grad()

	loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

	loss.backward()
	optimizer.step()

	del model.rewards[:]
	del model.saved_actions[:]

def main():
	running_reward = 10

	for i_episode in count(1):

		state = env.reset()
		ep_reward = 0

		for t in range(1, 10000):

			action = select_action(state)

			state, reward, done, _ = env.step(action)

			if args.render and i_episode % log_interval == 0:
				env.render()

			model.rewards.append(reward)
			ep_reward += reward
			if done:
				break

		running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

		finish_episode()

		if i_episode % log_interval == 0:
			print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
				  i_episode, ep_reward, running_reward))
			x_data.append(i_episode)
			y_data.append(running_reward)

		if running_reward > env.spec.reward_threshold:
			print("Solved! Running reward is now {} and "
				  "the last episode runs to {} time steps!".format(running_reward, t))
			break
		plot_figure(2, 'training','episodes','rewards', x_data, y_data)


if __name__ == '__main__':
	main()