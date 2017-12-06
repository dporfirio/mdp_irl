import xml.etree.ElementTree as ET
from MDP import State, Action

class UppaalReader():

	def __init__(self, file):
		self.file = file

	def parse(self):
		tree = ET.parse(self.file)
		root = tree.getroot()

		actiondict = {}
		statedict = {}

		for st in root.iter('location'):
			st_id = st.find('id')
			state = State(st_id)
			statedict[st_id] = state

		for a in root.iter('branchpoint'):
			a_id = a.find('id')
			action = Action(a_id)
			actiondict[a_id] = action

		for t in root.iter('transition'):
			pass

		actionlist = list(actiondict.keys())
		statelist = list(statedict.keys())

		return statelist,actionlist

class Dot():

	def __init__(self):
		pass

	#, fillcolor = "#ff0000A0", style=filled
	def cleanDot(self, dotfile, rewards, names=None):

		reward_dict = {}
		for i in range(0,len(rewards)):
			reward_dict[i] = rewards[i]

		with open(dotfile, "r") as infile:
			with open(dotfile[:-4] + "_modified.dot", "w") as outfile:

				action_name = None
				for line in infile:

					##### ~~~modifications~~~ #####

					# color the nodes appropriately
					if " [label=\"" in line:
						rew = reward_dict[int(line[0:line.index(" ")])]
						if rew < 0:
							color = "#ff0000"
						else:
							color = "#00ff00"

						alpha_raw = abs(rew * 255)
						if alpha_raw > 255:
							alpha_raw = 255

						alpha = hex(abs(int(round(alpha_raw))))
						alpha = alpha[2:]
						if len(alpha) == 1:
							alpha = "0" + alpha

						line = line[:-3] + ", fillcolor=\"" + color + alpha + "\", style=filled ];\n"

					# rename the nodes as needed

					# write modified line to outfile
					outfile.write(line)