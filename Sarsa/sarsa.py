# %matplotlib inline

import gym
import itertools
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys
import sklearn.pipeline
import sklearn.preprocessing

import warnings
import plotting
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler


matplotlib.style.use('ggplot')

warnings.filterwarnings("ignore", category=DeprecationWarning)

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
	from IPython import display

env = gym.envs.make('MountainCar-v0')

env.observation_space.sample()

observation_examples = np.array([env.observation_space.sample() for x in range(10000)])

scaler = sklearn.preprocessing.StandardScaler()

featurizer = sklearn.pipeline.FeatureUnion([
		('rbf1', RBFSampler(gamma = 5.0, n_components = 100)),
		('rbf2', RBFSampler(gamma = 2.0, n_components = 100)),
		('rbf3', RBFSampler(gamma = 1.0, n_components = 100)),
		('rbf4', RBFSampler(gamma = 0.5, n_components = 100))
		])

featurizer.fit(scaler.fit_transform(observation_examples))


class FunctionApproximator():

	def __init__(self):

		self.models = []
		for i in range(env.action_space.n):
			model = SGDRegressor(learning_rate = "constant")
			model.partial_fit([self.featurize_state(env.reset())], [0])
			self.models.append(model)

	def featurize_state(self, state):

		scaled = scaler.transform([state])
		features = featurizer.transform(scaled)
		return features[0]

	def predict(self, s, a = None):

		state_features = self.featurize_state(s)
		if not a:
			return np.array([m.predict([state_features])[0] for m in self.models])
		else:
			self.models[a].predict([state_features])[0]

	def update(self, s, a, y):

		features = self.featurize_state(s)
		self.models[a].partial_fit([features], [y])



def make_epsilon_greedy_policy(estimator, epsilon, nA):

	def policy_fn(observation):

		A = np.ones(nA, dtype = float) * epsilon / nA
		q_values = estimator.predict(observation)
		best_action = np.argmax(q_values)
		A[best_action] += 1 - epsilon
		return A
	return policy_fn


def sarsa(env, estimator, num_episodes, discount_factor = 1.0, epsilon = 0.1, epsilon_decay = 1.0):

	stats = plotting.EpisodeStats(
		episode_lengths = np.zeros(num_episodes),
		episode_rewards = np.zeros(num_episodes))

	for i_episode in range(num_episodes):

		policy = make_epsilon_greedy_policy(estimator, epsilon * epsilon_decay**i_episode, env.action_space.n)
		state = env.reset()
		action_probs = policy(state)
		action = np.random.choice(np.arange(len(action_probs)), p = action_probs)

		for t in itertools.count():

			next_state, reward, end, _ = env.step(action)
			next_action_probs = policy(next_state)
			next_action = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)

			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t

			q_values_next = estimator.predict(next_state)
			td_target = reward + discount_factor * q_values_next[next_action]

			estimator.update(state, action, td_target)

			if i_episode % 10 == 0:
				print("\rStep {} @ Episode {}/{} ({})".format(t, i_episode + 1, num_episodes, reward))
			
			if end:
				break

			state = next_state
			action = next_action
	return stats

estimator = FunctionApproximator()
stats = sarsa(env, estimator, 200, epsilon = 0.0)

plotting.plot_cost_to_go_mountain_car(env, estimator)
plotting.plot_episode_stats(stats, smoothing_window = 25)