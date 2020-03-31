'''
Monte Carlo Tree Search
Basic template
Functions will continue being added
'''

import numpy as np
import pandas as pd
import copy


def monte_carlo_search(root):

	while allow_search() == 1:

		leaf = traverse(copy.copy(root))
		sim_result = rollout(leaf)
		backpropogate(leaf, sim_result)

	return best_child(root)

def traverse(root):

	while len(root.children) > 0:
		root = best_child(root)
	return root

def best_child(root):

	temp = root.children[0]
	for i in range(len(root.children)):
		if temp.value < root.children[i].value:
			temp = root.children[i]

	return temp

def rollout(root):

	state = root.root
	action = np.argmax(estimater(state))
	next_state = env_model.step(action)
	next_state_node = Node(next_state, curr_id, root.id)
	root.expand(next_state_node)
	root.value = simulate(next_state_node)
	return root.value


def simulate(root):

	Gt = 0
	n_sims = 0
	env.set_state(root)
	state = root.state
	done = 0
	for i in range(np.randint(5,10)):

		while !done:

			state, reward, done, _ = env.step(policy(state))
			Gt += reward
		n_sims += 1
	return Gt/n_sims

def backpropogate(leaf, sim_result):

	par_node = leaf.parent
	par_node.value = (par_node.value * par_node.visits + sim_result) / (par_node.visits+1)
	par_node.visits +=1
	if par_node.parent != None:
		backpropogate(par_node, par_node.value)

class Node():

	def __init__(state, parent, node_id, parent_id):

		self.root = state
		self.id = node_id
		self.children = []
		self.value = 0
		self.wins = 0
		self.lose = 0
		self.visits = 0
		self.parent = parent
		self.parent_id = parent_id

	def expand(child):

		self.children.append(child)

	def sample():

		return np.random.choice(self.children, 1)[0]
