import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import linprog
from gurobipy import *
import random

class MDP():

	def __init__(self, statelist=None, actionlist=None):
		self.statelist = statelist
		self.actionlist = actionlist

		if statelist != None and actionlist != None:
			self.mdp = np.zeros((len(actionlist), len(statelist), len(statelist)))
			self.action_idx = {}
			for i in range(0,len(actionlist)):
				self.action_idx[actionlist[i]] = i
			state_idx = {}
			for i in range(0, len(statelist)):
				self.state_idx[statelist[i]] = i
		elif False: # if numpy file exists 
			pass
		else:
			self.generateFakeData()

	def generateFakeData(self):
		self.st1 = State("init")
		st2 = State("final")
		self.statelist = [self.st1, st2]

		a1 = Action("go")
		a2 = Action("goback")
		self.actionlist = [a1, a2]

		self.mdp = np.zeros((2,2,2))
		self.state_idx = {self.st1: 0, st2: 1}
		self.action_idx = {a1: 0, a2: 1}

		self.mdp[0,0,0] = 0.5
		self.mdp[0,0,1] = 0.5
		self.mdp[0,1,1] = 1.0
		self.mdp[1,1,0] = 1.0
		self.mdp[1,0,0] = 1.0

		print(self.mdp)

	def simulate_noninteractive(self, iterations=10):
		curr = self.state_idx[self.st1]

		for i in range(0,iterations):
			
			# select random action
			action = random.choice(self.actionlist)

			# obtain the rpw for that action and the current state
			row = self.mdp[self.action_idx[action]][curr]

			# obtain the next state
			choice = np.random.choice(self.statelist, p=row)
			print("action: {}, choice: {}".format(action,choice))

class State():

	def __init__(self, st_id):
		self.st_id = st_id

	def __str__(self):
		return str(self.st_id)

class Action():

	def __init__(self, a_id):
		self.a_id = a_id

	def __str__(self):
		return str(self.a_id)

class IRL():

	def __init__(self, mdp, R=None, gamma=0.9):
		self.R=np.ones(len(mdp.statelist))
		self.mdp = mdp.mdp
		self.statelist = mdp.statelist
		self.gamma = gamma

	def gurobi_test(self):
		try:

			# Create a new model
			m = Model("mip1")

			# Create variables
			x = m.addVar(vtype=GRB.BINARY, name="x")
			y = m.addVar(vtype=GRB.BINARY, name="y")
			z = m.addVar(vtype=GRB.BINARY, name="z")

			# Set objective
			m.setObjective(x + y + 2 * z, GRB.MAXIMIZE)

			# Add constraint: x + 2 y + 3 z <= 4
			m.addConstr(x + 2 * y + 3 * z <= 4, "c0")

			# Add constraint: x + y >= 1
			m.addConstr(x + y >= 1, "c1")

			m.optimize()

			for v in m.getVars():
				print('%s %g' % (v.varName, v.x))

			print('Obj: %g' % m.objVal)

		except GurobiError as e:
			print('Error code ' + str(e.errno) + ": " + str(e))

		except AttributeError:
			print('Encountered an attribute error')

	def ng_russel(self, r_max, lam):
		try:

			# create a new model
			m = Model("lp1")

			# create variables -- this part will likely take the longest
			maximin_vars = pd.Series(m.addVars(len(self.statelist)))  # this is the maximin vector
			r = pd.Series(m.addVars(len(self.statelist), ub=r_max)) # this is the reward vector

			# calculate the optimal policy matrix
			#self.mdp[0,0,0] = 0.5
			#self.mdp[0,0,1] = 0.5
			#self.mdp[0,1,1] = 1.0
			#self.mdp[1,1,0] = 1.0
			#self.mdp[1,0,0] = 1.0
			opt = np.zeros((2,2))
			opt[0] = [0.5,0.5]    # action 0
			opt[1] = [0.0,1.0]    # action 0
			opt_df = pd.DataFrame(opt)
			#mdp_df = pd.DataFrame(self.mdp)
			id_df = pd.DataFrame(np.identity(len(self.statelist)))

			# add constraints
			# z constraints
			#term1_1 = opt_df[0] - pd.DataFrame(self.mdp[0])[0]     # action 0, state 0
			#print(term1_1)
			term2 = pd.DataFrame(np.linalg.inv(np.identity(len(self.statelist)) - self.gamma*opt))
			#m.addConstr(maximin_vars[0] <= term1_1.dot(term2).dot(r), "c0")

			#term1_2 = opt_df[1] - pd.DataFrame(self.mdp[0])[1]     # action 0, state 1
			#print(term1_2)
			#m.addConstr(maximin_vars[0] <= term1_2.dot(term2).dot(r), "c1")

			term2_1 = pd.Series(opt[0] - self.mdp[1][0])#opt_df[0] - pd.DataFrame(self.mdp[1])[0]     # action 1, state 0
			#print(opt)
			#print(opt[0])
			#print(self.mdp[1][0])
			#print(term2_1)
			#print(opt_df[0] - pd.DataFrame(self.mdp[1])[0])
			#print(term2)
			#m.addConstr(maximin_vars[1] <= term2_1.dot(term2).dot(r), "c2")

			term2_2 =  pd.Series(opt[1] - self.mdp[1][1])#opt_df[1] - pd.DataFrame(self.mdp[1])[1]     # action 1, state 1
			#print(term2_2)
			#print(term1_2.dot(term2))
			#print(r)
			#print(term1_2.dot(term2).dot(r))
			#m.addConstr(maximin_vars[1] <= term2_2.dot(term2).dot(r), "c3")

			# irl constraints
			for i in range(0,2):
				term1 = pd.DataFrame(opt - self.mdp[i]) #opt_df - pd.DataFrame(self.mdp[i])
				print(term1)
				term2 = pd.DataFrame(np.linalg.inv(np.identity(len(self.statelist)) - self.gamma * opt))
				print(term2)
				#print("TERM 1: {}".format(term1))
				#print("TERM 2: {}".format(term2))
				#print(term1.dot(term2))
				#print(r)
				#print(term1.dot(term2).dot(r))
				m.addConstr(0 <= term1.dot(term2).dot(r)[0])
				m.addConstr(0 <= term1.dot(term2).dot(r)[1])

			# set objective
			obj = sum(maximin_vars) - lam*sum(r)
			m.setObjective(obj, GRB.MAXIMIZE)

			# optimize
			m.optimize()

			# print results
			print(m.objVal)
			#print(maximin_vars)
			print(r)



		except GurobiError as e:
			print('Error code ' + str(e.errno) + ": " + str(e))

		except AttributeError as e:
			print('Encountered an attribute error ' + str(e))


	# # # # # # # # # # # #
	# finite state spaces #
	# # # # # # # # # # # #
	def obj(self, x):
		suma = 0
		for i in range(0,len(self.statelist)):
			if i==0:
				pa1_mat = self.mdp[0]
				pmat = self.mdp[1]
			else:
				pa1_mat = self.mdp[1]
				pmat = self.mdp[0]

			suma += self.obj_helper(pa1_mat,pmat,i)

		return -1*(suma - self.gamma * np.linalg.norm(x))

	def obj_helper(self, Pa1, Pa, i):
		diff1 = Pa1[i] - Pa[i]
		diff2 = np.linalg.inv(np.identity(len(self.statelist)) - self.gamma*Pa1)

		prod1 = np.matmul(diff2,self.R)
		prod2 = np.matmul(diff1,prod1)

		return prod2

	def constraints(self, x):
		cons = []

		np.matmul((self.mdp[0]-self.mdp[1]),	 np.matmul(np.linalg.inv(np.identity(len(self.statelist)) - self.gamma*self.mdp[0]), x )  )

		return cons

	def constraint(self,x,i):
		pass

	def solve(self):
		#cons = ({'type': 'ineq',
		#		 'fun' : lambda x: np.matmul((self.mdp[0]-self.mdp[1]),	 np.matmul(np.linalg.inv(np.identity(len(self.statelist)) - self.gamma*self.mdp[0]), x )  )  })

		cons = ({'type': 'ineq',
				 'fun' : lambda x: -x+1})
		x=self.R
		print("solving...")
		res = minimize(self.obj, x, method='nelder-mead', options={'disp': True})
		print(res)

	def solve_test(self):
		x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
		res = minimize(self.rosen, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})

	def rosen(self, x):
		"""The Rosenbrock function"""
		return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)