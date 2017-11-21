import numpy as np
from scipy.optimize import maximize

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

	def __init__(mdp, R=np.zeros):
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