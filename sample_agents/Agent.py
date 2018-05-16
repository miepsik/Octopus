import random 
from array import array
import numpy as np

class Agent:
	"A template agent acting randomly"
	
	#name should contain only letters, digits, and underscores (not enforced by environment)
	__name = 'Python_Random'
	
	def __init__(self, stateDim, actionDim, agentParams):
		"Initialize agent assuming floating point state and action"
		self.__stateDim = stateDim
		self.__actionDim = actionDim
		self.__action = array('d',[0 for x in range(actionDim)])
		#we ignore agentParams because our agent does not need it.
		#agentParams could be a parameter file needed by the agent.
		random.seed()
		
		self.__epoch = 0
		self.__epsilon = 0.05
		self.__lastAction = None
		self.__lastState = None
		self.__alpha = 0.1
		self.__theta = np.zeros((self.__stateDim,self.__actionDim))
		
	   
	def __randomAction(self):
		self.__action = np.random.rand(self.__actionDim)
	
	def __curlAction(self):
		for i in range(self.__actionDim):
			if (i%3 == 2):
				self.__action[i] = 1
			else:
				self.__action[i] = 0
				
	def __supremeAction(self,state,reward):
		f = self.__lastFeatures
		#theta' = theta + alpha(R + gamma*max_a Q - Q)f_i
		# self.__theta += self.__alpha (self.__
		
			
	def start(self, state):
		"Given starting state, agent returns first action"
		self.__randomAction()
		
		self.__epoch += 1
		return self.__action
	
	def step(self, reward, state):
		"Given current reward and state, agent returns next action"
		self.__epoch += 1
		
		if random.random() <= self.__epsilon:
			self.__randomAction()
		else:
			self.__curlAction()
			# self.__supremeAction(reward,state)
		
		self.__lastAction = self.__action
		return self.__action
	
	def end(self, reward):
		pass
	
	def cleanup(self):
		pass
	
	def getName(self):
		return self.__name
	
			
