import numpy as np
from scipy.optimize import minimize

class MDP():

	def __init__(statelist, actionlist):
		mdp = np.zeros((len(actionlist), len(statelist), len(statelist)))
		action_idx = {}
		for i in range(0,len(actionlist)):
			action_idx[actionlist[i]] = i
		state_idx = {}
		for i in range(0, len(statelist)):
			state_idx[statelist[i]] = i



class State():

	def __init__():
		pass

class Action():

	def __init__():
		pass

class IRL():

	def __init__(self, mdp, R=np.zeros):
		pass

	# # # # # # # # # # # #
	# finite state spaces #
	# # # # # # # # # # # #
	def obj(x):
		pass

	def constraint():
		pass

	def solve():
		pass

	def solve_test(self):
		x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
		res = minimize(self.rosen, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})

	def rosen(self, x):
		"""The Rosenbrock function"""
		return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)