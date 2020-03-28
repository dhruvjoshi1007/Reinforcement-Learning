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

	plt.figure(fig_id)
	plt.clf()

	plt.title(fig_name)
	plt.xlabel(xlabel)
	plt.ylabel(y_label)

	plt.plot(x,y)

	plt.show()