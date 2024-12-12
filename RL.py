from MDP import build_mazeMDP, print_policy
import numpy as np
import time
import matplotlib.pyplot as plt


class ReinforcementLearning:
	def __init__(self, mdp, env):
		"""
		Constructor for the RL class

		:param mdp: Markov decision process (T, R, discount)
		:param sampleReward: Function to sample rewards (e.g., bernoulli, Gaussian). This function takes one argument:
		the mean of the distribution and returns a sample from the distribution.
		"""

		self.mdp = mdp
		self.env = env

	def sampleRewardAndNextState(self,state,action):
		'''Procedure to sample a reward and the next state
		reward ~ Pr(r)
		nextState ~ Pr(s'|s,a)

		Inputs:
		state -- current state
		action -- action to be executed

		Outputs:
		reward -- sampled reward
		nextState -- sampled next state
		'''

		reward = self.sampleReward(self.mdp.R[action,state])
		cumProb = np.cumsum(self.mdp.T[action,state,:])
		nextState = np.where(cumProb >= np.random.rand(1))[0][0]
		return [reward,nextState]
	
	def OffPolicyTD(self, nEpisodes, epsilon=0.0):
		'''
		Off-policy TD (Q-learning) algorithm
		Inputs:
		nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0)
		epsilon -- probability with which an action is chosen at random
		Outputs:
		Q -- final Q function (|A|x|S| array)
		policy -- final policy
		'''
		nActions = self.mdp.nActions
		nStates = self.mdp.nStates
		discount = self.mdp.discount

		# Initialize Q-function and policy
		Q = np.zeros((nActions, nStates))  # Q-values for state-action pairs
		policy = np.zeros(nStates, int)   # Greedy policy (deterministic)
		cumulate_rewards = [] 

		for _ in range(nEpisodes):
			state = 0
			i = 0 # Initialize step counter
			cumulate_reward = 0
			gamma_k = 1 # gamma^k

			while True:
				i += 1
				alpha = 5 / ( i + 5 )  # Decreasing learning rate
				action = self.behavior_policy(state, Q, epsilon)
				reward, nextState = self.sampleRewardAndNextState(state, action)

				gamma_k *= discount
				cumulate_reward += gamma_k * reward

				best_next_action = np.argmax(Q[:, nextState])  # Best action under current Q
				Q[action, state] += alpha * (reward + discount * Q[best_next_action, nextState] - Q[action, state])

				policy[state] = np.argmax(Q[:, state])
				state = nextState

				if state == 16:  
					break
			cumulate_rewards.append(cumulate_reward)
		
		return [Q, policy, cumulate_rewards]


	# Behavior policy (epsilon-soft)
	def behavior_policy(self, state, Q, epsilon):
		if np.random.rand() < epsilon:
			return np.random.randint(self.mdp.nActions)  
		else:
			return Q[:, state].argmax()
		
	def construct_trajectory(self, Q, epsilon, max_length):
		trajectory = []
		length = 0
		state = 0 # initial state
		while length < max_length:
			length += 1
			action = self.behavior_policy(state, Q, epsilon)
			[reward, next_state] = self.sampleRewardAndNextState(state, action)
			trajectory.append((state, action, reward))
			
			state = next_state
		return trajectory

	def OffPolicyMC(self, nEpisodes, epsilon=0.0):
		'''
		Off-policy MC algorithm with epsilon-soft behavior policy
		Inputs:
		nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
		epsilon -- probability with which an action is chosen at random
		Outputs:
		Q -- final Q function (|A|x|S| array)
		policy -- final policy
		'''
		# temporary values to ensure that the code compiles until this
		# function is coded
		nActions = self.mdp.nActions
		nStates = self.mdp.nStates
		discount = self.mdp.discount

		# Initialize Q-function and C (for importance sampling corrections)
		Q = np.zeros((nActions, nStates))
		C = np.zeros((nActions, nStates))  
		policy = np.zeros(nStates, int)  
		Max_b_len = 20  # Maximum length of a trajectory
		cumulate_rewards = []

		for _ in range(nEpisodes):
			state = 0  # initial state
			trajectory = self.construct_trajectory(Q, epsilon, Max_b_len)
			
			G = 0  
			W = 1  
			for state, action, reward in reversed(trajectory):
				G = discount * G + reward  
				C[action, state] += W
				Q[action, state] += W / C[action, state] * (G - Q[action, state])
				policy[state] = np.argmax(Q[:, state])  

				if policy[state] != action:
					break
				W *= 1.0 / (1.0 - epsilon + epsilon / nActions)  # Adjust weight for epsilon-soft policy
			cumulate_rewards.append(G)
		return [Q, policy, cumulate_rewards]

def compute_average_rewards(cumulative_rewards, group_size=20):
    # Group rewards by averaging over `group_size` episodes
    avg_rewards = [
        np.mean(cumulative_rewards[i:i + group_size])
        for i in range(0, len(cumulative_rewards), group_size)
    ]
    return avg_rewards

if __name__ == '__main__':
	mdp = build_mazeMDP()
	rl = ReinforcementLearning(mdp, np.random.normal)
	nepisodes = 4000
	epsilon = 0.1
	group_size = 20

	# # Test Q-learning
	
	rl = ReinforcementLearning(mdp, np.random.normal)
	plots_dict = {}
	for epsilon in [0.05, 0.1, 0.3, 0.5]:
		[Q, policy, cumulate_rewards] = rl.OffPolicyTD(nEpisodes=nepisodes, epsilon=epsilon)
		print_policy(policy)
		plots_dict[epsilon] = cumulate_rewards
		print(epsilon, np.mean(cumulate_rewards))


	plt.figure(figsize=(10, 6))
	for epsilon, cumulate_rewards in plots_dict.items():
		avg_rewards = compute_average_rewards(cumulate_rewards, group_size)
		x_values = range(group_size, nepisodes + 1, group_size)
		plt.plot(x_values, avg_rewards, label=f'epsilon = {epsilon}')
	plt.title(f'OffPolicyTD : Cumulative Reward vs. Episodes (Averaged every {group_size} episodes)')
	plt.xlabel(f'Episode Group ({group_size} episodes each)')
	plt.ylabel('Average Cumulative Reward')
	plt.legend(fontsize=12, loc="lower right")
	plt.grid()
	plt.savefig('TD.png')
	plt.show()

	# Test Off-Policy MC
	# [Q, policy, cumulate_rewards] = rl.OffPolicyMC(nEpisodes=nepisodes, epsilon=epsilon)
	# print_policy(policy)
	# plt.close()
	# plt.figure(figsize=(10, 6))
	# avg_rewards = compute_average_rewards(cumulate_rewards, group_size)
	# x_values = range(group_size, nepisodes + 1, group_size)
	# plt.plot(x_values, avg_rewards, label=f'epsilon = {epsilon}')
	# plt.title(f'OffPolicyMC : Cumulative Reward vs. Episodes (Averaged every {group_size} episodes)')
	# plt.xlabel(f'Episode Group ({group_size} episodes each)')
	# plt.ylabel('Average Cumulative Reward')
	# plt.grid()
	# plt.savefig('MC.png')
	# plt.show()

	# plt.close()
	# plt.figure(figsize=(10, 6))
	# plt.plot(range(len(cumulate_rewards)), cumulate_rewards, label=f'epsilon = {epsilon}')
	# plt.title(f'OffPolicyMC : Cumulative Reward vs. Episodes (Averaged every {group_size} episodes)')
	# plt.xlabel(f'Episode Group ({group_size} episodes each)')
	# plt.ylabel('Average Cumulative Reward')
	# plt.grid()
	# plt.savefig('MC.png')
	# plt.show()