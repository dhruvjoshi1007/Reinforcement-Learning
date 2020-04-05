'''
PyTorch implementation of reinforce
'''

import argparse
import gym
import numpy as np
from itertools import count

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
		self.layer1 = nn.Linear(4,128)
		self.drop = nn.Dropout(p=0.6)
		self.layer2 = nn.Linear(128,2)

		self.saved_log_probs = []
		self.rewards = []

	def forward(self, x):

		x = self.layer1(x)
		x = self.drop(x)
		x = F.relu(x)
		action_scores = self.layer2(x)

		return F.softmax(action_scores, dim = 1) 


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr = 1e-2)
eps = np.finfo(np.float32).eps.item()
print('---------------------------------------------------')

def select_action(state):

	state = torch.from_numpy(state).float().unsqueeze(0)
	probs = policy(state)
	m = Categorical(probs)
	action = m.sample()
	policy.saved_log_probs.append(m.log_prob(action))

	return action.item()


def finish_episode():
	
	R = 0
	policy_loss = []
	returns = []
	for r in policy.rewards[::-1]:
		R = r + gamma*R
		returns.insert(0, R)
	returns = torch.tensor(returns)
	returns = (returns - returns.mean()) / (returns.std() + eps)

	for log_prob, R in zip(policy.saved_log_probs, returns):

		policy_loss.append(-log_prob * R)

	optimizer.zero_grad()
	policy_loss = torch.cat(policy_loss).sum()
	policy_loss.backward()
	optimizer.step()
	del policy.rewards[:]
	del policy.saved_log_probs[:]


def main():

	running_reward = 10
	for i_episode in count(1):
		state, ep_reward = env.reset(), 0
		for t in range(10000):

			action = select_action(state)
			state, reward, done, _ = env.step(action)
			if args.render:
				env.render()
			policy.rewards.append(reward)
			ep_reward += reward
			if done:
				break

		running_reward = 0.05 * ep_reward +(1-0.05) * running_reward
		x_data.append(i_episode)
		y_data.append(running_reward)
		plot_figure(2, 'training','episodes','rewards', x_data, y_data)
		finish_episode()
		if i_episode % log_interval == 0:
			print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
				i_episode, ep_reward, running_reward))
		if running_reward > env.spec.reward_threshold:
			print("Solved! Running reward is now {} and "
				"the last episode runs to {} time steps!".format(running_reward, t))
			break


if __name__ == '__main__':
	main()
	# plt.ioff()
	plt.show()



