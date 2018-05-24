# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, InputLayer
import math

import os


# p1 is the center point
def angleBetweenPoints(p0, p1, p2):
    a = (p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2
    b = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
    c = (p2[0] - p0[0]) ** 2 + (p2[1] - p0[1]) ** 2
    return math.acos((a + b - c) / math.sqrt(4 * a * b))


class Agent:
    "A template agent learning using Tensorflow"

    # name should contain only letters, digits, and underscores (not enforced by environment)
    __name = 'TF1'

    def __init__(self, stateDim, actionDim, agentParams):
        "Initialize agent assuming floating point state and action"

        self.featureExtractors = [getattr(self, m) for m in dir(self) if
                                  m.startswith("_Agent__extractFeature") and callable(getattr(self, m))]

        self.__realStateDim = stateDim
        self.__stateDim = len(self.featureExtractors)
        self.__realActionDim = actionDim
        self.__actionDim = 3
        assert self.__actionDim % 3 == 0
        assert (stateDim - 2) % 4 == 0
        self.__action = np.zeros(actionDim)

        # Set up control variables
        self.alpha = 0.1
        self.g = 0.9
        self.e = 0.1
        self.__step = 0
        self.angle = np.arctan2(9, -1)
        self.model = keras.models.load_model('model2')

        self.__reward = 0

    def __extractFeatureReward(self, *args):
        "Reward"
        return self.__reward

    def __extractFeatureCenterOfGravity(self, *args):
        "Center of gravity (equal masses of points)"
        state = args[0].copy()
        xy = state[0, 2:].reshape(int((self.__realStateDim - 2) / 4), 4)[:, :2]
        return np.sqrt(np.square(xy).sum(1)).mean()

    def __extractFeatureDistance(self, *args):
        "Euklidean distance from closest to (9,-1)"
        state = args[0].copy()
        xy = state[0, 2:].reshape(int((self.__realStateDim - 2) / 4), 4)[:,
             :2]  # lista punktów (x,y)(czubki segmentów macki)
        xy -= [9, -1]
        return np.sqrt(np.square(xy).sum(1).min())

    def __extractFeatureAngleParallelity(self, *args):
        state = args[0][0]
        "Angle stopping tentacle from parallelity with [(0,0),(9,-1)]"
        return angleBetweenPoints((9, -1), state[34:36], state[38:40])

    def __extractFeatureAngleParallelity2(self, *args):
        state = args[0][0]
        "Angle stopping tentacle from parallelity with [(0,0),(9,-1)]"
        return angleBetweenPoints((9, -1), state[74:76], state[78:80])

    def __extractFeatureVertexCloser(self, *args):
        "Which vertex of the tentacle is closer (-1 if lower, 1 if upper)"
        state = args[0].copy()
        xy = state[0, 2:].reshape(int((self.__realStateDim - 2) / 4), 4)[[9, 19],
             :2]  # lista punktów (x,y)(czubki segmentów macki)
        return np.argmin(np.square(xy - [9, -1]).sum(1)) * 2 - 1

    def __extractImportantPoints(self, *args):
        state = args[0][0][2:]
        l = []
        for i in (0, 4, 5, 7, 8, 9):
            for j in range(4):
                l.append(state[4 * i + j])
            for j in range(4):
                l.append(state[4 * i + 40 + j])
        for i in range(12):
            l[i * 4] -= 4.5
            l[i * 4 + 1] -= 1
        return l

    def getFeatureVector(self, state, reward):
        "Convert input parameters to vecture of features"
        self.__reward += reward
        st = state.copy()
        state = np.array(list(state)).reshape((1, self.__realStateDim))
        f = []
        for m in self.featureExtractors:
            s = state.copy()
            # print(m.__name__,m(state,reward))
            f += [m(s, reward)]
        f += self.__extractImportantPoints(st)
        return f

    def __getActionAndItsPrediction(self, state):
        "Choose an action by greedily (with e chance of random action) from the Q-network"

        a, allQ = self.__session.run([self.bestQ, self.Q], feed_dict={self.input: state})

        if np.random.rand(1) < self.e:
            a = np.random.randint(0, 2 ** self.__actionDim, size=(1))

        self.__lastState = state
        self.__actionEncoded = a[0]
        self.__predictedQ = allQ[0, a[0]]
        self.__predictedAllQ = allQ
        return self.__decodeAction(a[0])

    def __getReward(self, state, reward):
        "distance from point + reward"
        self.__reward = 8 - self.__extractFeatureDistance(state) / 9.05538 + reward - \
                        abs(self.__extractFeatureAngleParallelity(state)) - \
                        abs(self.__extractFeatureAngleParallelity2(state))

    def start(self, state):
        "Given starting state, agent returns first action"
        return self.step(0, state)

    def step(self, reward, state):
        "Given current reward and state, agent returns next action"
        self.__step += 1
        self.__reward += reward
        state = np.array(list(state)).reshape((1, self.__realStateDim))
        l = []
        self.__getReward(state, reward)
        l.append(self.getFeatureVector(state, reward))
        target = self.model.predict(np.array(l))

        self.__action = self.__decodeAction(np.argmax(target))

        # Reduce chance of random action as we train the model.
        return self.__action

    def end(self, reward):
        pass

    def cleanup(self):
        pass

    def getName(self):
        return self.__name

    def __decodeAction(self, a):
        x = []
        for i in range(3):
            y = a % 6
            for j in range((5, 4, 1)[i]):
                x += {0: [0, 0, 0], 1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 1], 4: [1, 1, 0], 5: [0, 1, 1]}[y]
            a //= 6
        return x

    def __destruct(self):
        print("Ala ma kota")

    def __exit__(self, exc_type, exc_value, traceback):
        self.__destruct()

    def __del__(self):
        self.__destruct()
