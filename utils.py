'''
Utility functions

** will be adding plotting functions and network utility functions in the future
'''


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import argparse


is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
	from IPython import display

parser = argparse.ArgumentParser(description='args')

parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')

parser.add_argument('--render', action='store_true',
					help='render the environment')
args = parser.parse_args()


def greedy(q_values):

	'''
	Function for greedy policy during inferance
	'''

	return np.argmax(q_values)

def epsilon_greedy(epsilon, q_values):

	'''
	Function for epsilon greedy policy during training
	Value iteration methods
	'''

	nA = len(q_values)
	action_probs = np.ones(nA, dtype = float) * epsilon / nA
	best_action = np.argmax(q_values)
	action_probs[best_action] += 1 - epsilon

	action = np.random.choice(np.arange(nA), p = action_probs)

	return action

def epsilon_greedy_policy(epsilon, action_probs):

	'''
	Function for epsilon greedy policy
	Policy gradient methods
	'''

	return np.random.choice(np.arange(len(action_probs)), p = action_probs)


def plot_figure(fig_id, fig_name, x_label, y_label, x, y):

	'''
	Function for plotting figures
	'''

	plt.figure(fig_id)
	plt.clf()

	durations_t = torch.tensor(1, dtype = torch.float)

	plt.title(fig_name)
	plt.xlabel(xlabel)
	plt.ylabel(y_label)

	plt.plot(x,y)

	plt.pause(0.001)

	if is_ipython:
		display.clear_output(wait=True)
		display.display(plt.gcf())



class ReplayMemory(object):

	'''
	Replay Memory class to allow storing transitions
	'''

	def __init__(self, capacity):

		self.capacity = capacity
		self.memory = []
		self.position = 0

	def push(self, item):

		if len(self.memory) < self.capacity:
			self.memory.append(None)
		
		self.memory[self.position] = item
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch):

		return random.sample(self.memory, batch)

	def __len__(self):

		return len(self.memory)


class NeuralNetwork(nn.Module):

	def __init__(self, nA, input_dim):

		super().__init__()

		self.layer1 = nn.Linear(input_dim, input_dim)
		self.bn1 = nn.BatchNorm2d(input_dim)
		self.layer2 = nn.Linear(input_dim, input_dim//2)
		self.bn1 = nn.BatchNorm2d(input_dim//2)
		self.layer3 = nn.Linear(input_dim//2, input_dim//4)
		self.bn1 = nn.BatchNorm2d(input_dim//4)

		self.output = nn.Linear(input_dim//4, nA)

	def forward(self, x):

		x = F.relu(self.bn1(self.layer1(x)))
		x = F.relu(self.bn2(self.layer2(x)))
		x = F.relu(self.bn3(self.layer3(x)))

		return self.output(x)


