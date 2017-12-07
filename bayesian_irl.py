import numpy as np
import random
import copy
import sys

class BayesianIRL():

	def __init__(self, mdp, gamma, opt, T=1000):
		self.T=T
		self.mdp=mdp
		self.gamma=gamma
		self.opt=opt

	def irl(self, delta, reward=None):
		#1 pick a random reward vector
		self.n_states = len(self.mdp)
		self.n_actions = len(self.mdp[0])
		if reward == None:
			x = np.linspace(-1,1,2/delta+1)
			reward = np.random.choice(x,self.n_states)
		#print(reward)

		#2 initial policy iteration
		values,pi = self.policy_iteration(reward)

		#print(reward)
		#print(values)
		#print(pi)

		#3 repeat
		t = 0
		Q = np.zeros((self.n_states,self.n_actions))
		Q_old = np.zeros((self.n_states,self.n_actions))

		# visualization variables
		acceptance_probabilities = np.zeros(self.T)
		while t<self.T:

			# a) pick a reward vector uniformly at random
			idx = np.random.randint(len(reward))
			sign = np.random.choice([-1,1], 1)
			#print("{} sign: ".format(sign))
			new_reward = reward.copy()

			while True:
				newval = new_reward[idx] + sign*delta
				newval = int(round(newval[0]*100))/100.0    # TODO: bound this by R-max (abs value)

				if newval <= 1.0 and newval >= -1.0:
					new_reward[idx] = newval
					break

				idx = np.random.randint(len(reward))
				sign = np.random.choice([-1,1], 1)

			# b) compute the new Q
			exists_worse = False

			# new way of computing Q!
			for act in range(0,self.n_actions):
				# compute the coefficient matrix
				ident = np.identity(self.n_states)
				T = np.zeros((self.n_states,self.n_states))
				for st_src in range(0,self.n_states):
					for st_targ in range(0,self.n_states):
						T[st_src][st_targ] = self.mdp[st_src][act][st_targ]
				
				coef_matrix = np.linalg.inv(ident - self.gamma * T)

				for st in range(0, self.n_states):
					Q[st][act] = np.inner(coef_matrix[st],new_reward)
					Q_old[st][act] = np.inner(coef_matrix[st],reward)

				# c-1 check if there exists an action better than the CURRENT optimal policy
			for st in range(0, self.n_states):
				if Q[st][pi[st]] < np.max(Q[st]):
					exists_worse = True
					#print("{} THERE EXISTS WORSE".format(t))
					#print(Q[st])
					#print(Q[st][pi[st]])
					#print(values)
					#print(reward)
					#print()
			#print(Q)

			# c-2) check if there exists an (s,a) s.t. the policy is worse
			alpha = 1
			# first the energies
			E_new = 0
			E_old = 0

			for st in range(0,self.n_states):
				E_new += Q[st][self.opt[st]]
				E_old += Q_old[st][self.opt[st]]

			# compute the normalization constants
			Z_new = 0
			Z_old = 0

			for st in range(0,self.n_states):
				Z_new += np.exp(Q[st][self.opt[st]])
				Z_old += np.exp(Q_old[st][self.opt[st]])

			Z_new = 1.0/pow(Z_new, self.n_states)
			Z_old = 1.0/pow(Z_old, self.n_states)

			P_new = Z_new*np.exp(alpha * E_new)
			P_old = Z_old*np.exp(alpha * E_old)
			#print("p_new: {}, p_old: {}".format(P_new, P_old))
			acceptance = min(1, P_new/P_old)
			acceptance_probabilities[t] = acceptance

			if exists_worse:

				# i) update the policy
				new_values,new_pi = self.policy_iteration(new_reward,pi.copy())

				# b) calculate probability
				sample = np.random.uniform(0,1)
				if sample <= acceptance:
					pi = new_pi
					values = new_values
					reward = new_reward


			else:
				sample = np.random.uniform(0,1)
				if sample <= acceptance:
					reward = new_reward



			t += 1

		return reward, acceptance_probabilities

	def policy_iteration(self, reward, pi=None):
		#look at page 87 of http://ufal.mff.cuni.cz/~straka/courses/npfl114/2016/sutton-bookdraft2016sep.pdf
		# initialize if needed
		if pi is None:
			pi = self.get_random_policy()
		values = np.random.uniform(0,1,self.n_states)

		# iterations of policy evaluation, followed by policy improvement
		num_itr = 0
		num_policy_eval_itr = 0
		while True:
			num_itr += 1

			# policy evaluation
			theta = 0.0001
			while True:
				delta = 0
				num_policy_eval_itr += 1
				for st in range(0, self.n_states):
					v = values[st]

					#valsum = reward[st]
					#for st_1 in range(0,self.n_states):
					#	valsum += self.mdp[st][pi[st]][st_1] * (self.gamma*values[st_1])
					#values[st] = valsum
					values[st] = np.dot(self.mdp[st][pi[st]], reward + self.gamma*values)

					delta = max(delta, abs(v - values[st]))
				if delta < theta:
					break
		
			# policy improvement
			policy_stable = True
			for st in range(0, self.n_states):
				old_action = pi[st]
				#candidate_actions = np.zeros(4)
				#for act in range(0, self.n_actions):
				#	valsum = reward[st]
				#	for st_1 in range(0,self.n_states):
				#		valsum += self.mdp[st][act][st_1] * (self.gamma*values[st_1])
				#	candidate_actions[act] = valsum
				candidate_actions = np.dot(self.mdp[st], reward + self.gamma*values)
				pi[st] = np.argmax(candidate_actions)

				if old_action != pi[st]:
					policy_stable = False

			if policy_stable:
				break

		print("final pi: {}".format(pi))
		print("number of iterations of initial policy iteration: {}".format(num_itr))
		print("number of iterations of initial policy evaluation: {}".format(num_policy_eval_itr))
		return values, pi

	def get_random_policy(self):
		actions = {}
		for st in range(0, self.n_states):
			actions[st] = []
			for act in range(0, self.n_actions):
				if sum(self.mdp[st][act]) > 0:
					actions[st].append(act)

		pi = np.zeros(self.n_states)
		for key in actions:
			if len(actions[key]) > 0: 
				pi[key] = np.random.choice(np.array(actions[key]))

		return pi.astype(int)
