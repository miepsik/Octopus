import numpy as np
import tensorflow as tf

import os

class Agent:
	"A template agent learning using Tensorflow"
	
	#name should contain only letters, digits, and underscores (not enforced by environment)
	__name = 'TF0'
	
	def __init__(self, stateDim, actionDim, agentParams):
		"Initialize agent assuming floating point state and action"
		
		self.featureExtractors = [getattr(self, m) for m in dir(self) if m.startswith("_Agent__extractFeature") and callable(getattr(self, m))]
		
		self.__realStateDim = stateDim
		self.__stateDim = len(self.featureExtractors)
		self.__realActionDim = actionDim
		self.__actionDim = 3
		assert self.__actionDim%3==0
		assert (stateDim - 2)%4==0
		self.__action = np.zeros(actionDim)
		
		#Set up control variables
		self.alpha = 0.1
		self.g = 0.9
		self.e = 0.1
		self.__step = 0
		self.CKPT = "../../tmp/model.ckpt"
		self.angle = np.arctan2(9,-1)
		
		#Set up neural network
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
		tf.reset_default_graph()
		self.input = tf.placeholder(shape=[1,self.__stateDim],dtype=tf.float32)
		self.theta = tf.Variable(tf.random_uniform([self.__stateDim,2**self.__actionDim],0,0.01))
		self.Q	  = tf.matmul(self.input,self.theta)
		self.bestQ= tf.argmax(self.Q,1)
		
		self.Qprime= tf.placeholder(shape=[1,2**self.__actionDim],dtype=tf.float32)
		self.loss  = tf.reduce_sum(tf.square(self.Qprime - self.Q))
		self.trainer = tf.train.GradientDescentOptimizer(learning_rate=self.alpha)
		self.updateModel = self.trainer.minimize(self.loss)
		self.init = tf.global_variables_initializer()
		self.saver = tf.train.Saver()
		self.__session = tf.Session()
		self.__session.run(self.init)
		
		self.__reward = 0
		
		if os.path.isfile(self.CKPT):
			self.saver.restore(self.__session,self.CKPT)
			
	def __extractFeatureReward(self,*args):
		"Reward"
		return self.__reward
			
	def __extractFeatureCenterOfGravity(self,*args):
		"Center of gravity (equal masses of points)"
		state = args[0]
		xy = state[0,2:].reshape(int((self.__realStateDim-2)/4),4)[:,:2]
		return np.sqrt(np.square(xy).sum(1)).mean()
		
			
	def __extractFeatureDistance(self,*args):
		"Euklidean distance from closest to (9,-1)"
		state = args[0]
		xy = state[0,2:].reshape(int((self.__realStateDim-2)/4),4)[:,:2]
		xy-= [9,-1]
		return np.sqrt(np.square(xy).sum(1).min())
		
	def __extractFeatureAngleParallelity(self,*args):
		"Angle stopping tentacle from parallelity with [(0,0),(9,-1)]"
		return args[0][0][0] - self.angle
		#Nie wiem, czy kąt zwracany przez status nie jest aby przesunięty
		#Może się opłacać znormalizować a między <-pi,pi>
	
	def __extractFeatureVertexCloser(self,*args):
		"Which vertex of the tentacle is closer (-1 if lower, 1 if upper)"
		state = args[0]
		xy = state[0,2:].reshape(int((self.__realStateDim-2)/4),4)[[9,19],:2]
		return np.argmin(np.square(xy - [9,-1]).sum(1))*2-1	
		
	def __getFeatureVector(self,state,reward):
		"Convert input parameters to vecture of features"
		f = []
		for m in self.featureExtractors:
			# print(m.__name__,m(state,reward))
			f += [m(state,reward)]
		return np.array(f).reshape((1,self.__stateDim))
		
	def __getActionAndItsPrediction(self,state):
		"Choose an action by greedily (with e chance of random action) from the Q-network"
		
		a,allQ = self.__session.run([self.bestQ,self.Q],feed_dict={self.input:state})

		if np.random.rand(1) < self.e:
			a = np.random.randint(0,2**self.__actionDim,size=(1))
		
		self.__lastState = state
		self.__actionEncoded = a[0]
		self.__predictedQ = allQ[0,a[0]]
		self.__predictedAllQ = allQ
		return self.__decodeAction(a[0])
		
	def __getReward(self,state,reward):
		"distance from point + reward"
		self.__reward = 1 - self.__extractFeatureDistance(state)/9.05538 + reward
		
			
	def start(self, state):
		"Given starting state, agent returns first action"
		self.__step += 1
		state = np.array(list(state)).reshape((1,self.__realStateDim))
		state = self.__getFeatureVector(state,0)
		self.__action = self.__getActionAndItsPrediction(state)
		return self.__action
	
	def step(self, reward, state):
		"Given current reward and state, agent returns next action"
		self.__step += 1
		self.__reward += reward
		state = np.array(list(state)).reshape((1,self.__realStateDim))
		
		self.__getReward(state,reward)
		state = self.__getFeatureVector(state,reward)
		
		Q = self.__session.run(self.Q,feed_dict={self.input:state})
		maxQ = np.max(Q)
		targetQ = self.__predictedAllQ
		targetQ[0,self.__actionEncoded] = self.__reward + self.g * maxQ
		#Train our network using target and predicted Q values
		_,W1 = self.__session.run([self.updateModel,self.theta],feed_dict={self.input:self.__lastState,self.Qprime:targetQ})
		
		self.__action = self.__getActionAndItsPrediction(state)		
		
		#Reduce chance of random action as we train the model.
		self.e = 1./(self.__step/1000 + 10)
		
		if self.__step%1001 == 0 or reward==10:
			self.saver.save(self.__session, self.CKPT)
				
		return self.__action
	
	def end(self, reward):
		pass
	
	def cleanup(self):
		pass
	
	def getName(self):
		return self.__name
	
	def __decodeAction(self,a):
		b = np.array(list(bin(a))[2:])

		l = self.__actionDim - len(b)
		b = np.append(np.zeros(l),b).reshape((int(self.__actionDim/3),3))
		b = np.repeat(b,int(self.__realActionDim/self.__actionDim),axis=0)
		l = int(self.__realActionDim/3 - len(b))
		b = np.append(b,np.full((l,3),b[-1]))

		return b.reshape((self.__realActionDim)).astype(np.float32)
		
	def __destruct(self):
		print("Ala ma kota")
		self.__session.close()
		
	def __exit__(self, exc_type, exc_value, traceback):
		self.__destruct()
		
	def __del__(self):
		self.__destruct()
	
			
