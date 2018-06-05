# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, InputLayer
import math
import random

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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
        self.alpha = 0.9
        self.g = 0.99
        self.e = -0.2
        self.__step = 0
        self.angle = np.arctan2(9, -1)

        self.model = keras.models.load_model('model44')
        self.__reward = 0
        self.previousStep = np.array([])
        self.previousValues = np.array([])
        self.previousAction = 0

        self.strangeInput = []
        self.strangeDecision = []
        self.normalOut = []
        self.allInput = []
        self.allOut = []
        self.allDecision = []

        self.up = True

        self.version = 0

    # def __extractFeatureReward(self, *args):
    #     "Reward"
    #     return self.__reward
    #
    # def __extractFeatureCenterOfGravity(self, *args):
    #     "Center of gravity (equal masses of points)"
    #     state = args[0].copy()
    #     xy = state[0, 2:].reshape(int((self.__realStateDim - 2) / 4), 4)[:, :2]
    #     return np.sqrt(np.square(xy).sum(1)).mean()

    def learnFromAll(self):
        mypath = "data/"
        files = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
        x = []
        y = []
        for s in files:
            if s == "dat" + str(self.version):
                continue
            f = open(mypath + s, 'r')
            n = int(f.readline())
            inp = []
            # f.readline()
            for i in range(n):
                inp.append(np.array([float(x) for x in f.readline().split()]))
            out = []
            for i in range(n):
                out.append(np.array([float(x) for x in f.readline().split()]))
                for j in range(len(out[i])):
                    if out[i][j] >= 15 - 0.01 * n - 0.05 * (n - i):
                        out[i][j] = 13 - 0.01 * n - 0.05 * (n - i)
            # f.readline()
            for i in range(n):
                out[i][int(f.readline())] = 15 - 0.01 * n - 0.05 * (n - i)
                x.append(inp[i])
                y.append(out[i])
        x = np.array(x)
        y = np.array(y)
        self.model.fit(x, y, epochs=55, batch_size=1, verbose=1)

    def learnFromFile(self, file):
        if file == "data/dat" + str(self.version):
            return
        with open(file, "r") as f:
            n = int(f.readline())
            inp = []
            # f.readline()
            for i in range(n):
                inp.append(np.array([float(x) for x in f.readline().split()]))
            out = []
            for i in range(n):
                out.append(np.array([float(x) for x in f.readline().split()]))
                for j in range(len(out[i])):
                    if out[i][j] >= 15 - 0.01 * n - 0.05 * (n - i):
                        out[i][j] = 13 - 0.01 * n - 0.05 * (n - i)
            # f.readline()
            for i in range(n):
                out[i][int(f.readline())] = 15 - 0.01 * n - 0.05 * (n - i)
            x = np.array(inp)
            y = np.array(out)
            self.model.fit(np.array(inp), np.array(out), epochs=5, batch_size=1, verbose=1)

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

    def shouldAttack(self, state, orgState):
        orgState = orgState[0]
        angle1 = angleBetweenPoints((9, -1), orgState[26:28], orgState[30:32])
        angle2 = angleBetweenPoints((9, -1), orgState[66:68], orgState[70:72])
        # print(state[1], angle1, state[2], angle2, state[0])
        if (state[1] + angle1 < 0.5 or state[2] + angle2 < 0.5) and state[0] < .25:
            return True
        return False

    def __extractImportantPoints(self, *args):
        state = args[0][0][2:]
        l = []
        for i in (0, 4, 7):
            for j in range(4):
                l.append(state[4 * i + j])
            for j in range(4):
                l.append(state[4 * i + 40 + j])
        for i in range(6):
            l[i * 4] -= 4.5
            l[i * 4 + 1] -= 1
        return l

    def __extractUpOrLow(self, state):
        xy = np.array(state[0, 2:]).reshape(int((self.__realStateDim - 1) / 4), 4)[:, :2]
        a = -1 / 9
        b = 0
        ul = xy[:, 0] * a + b - xy[:, 1]
        # print(((ul<0).sum() - (ul>0).sum())>0)
        return ((ul < 0).sum() - (ul > 0).sum()) > 0

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
        self.__reward = 3 - 2 * self.__extractFeatureDistance(state) / 9.05538 + reward - \
                        abs(self.__extractFeatureAngleParallelity(state)) - \
                        abs(self.__extractFeatureAngleParallelity2(state))

    def start(self, state):
        "Given starting state, agent returns first action"
        self.alpha = 0.9
        self.g = 0.99
        self.e = 0.1
        self.__step = 0
        self.__reward = 0
        self.previousStep = np.array([])
        self.previousValues = np.array([])
        self.previousAction = 0

        self.strangeInput = []
        self.strangeDecision = []
        self.normalOut = []
        self.allInput = []
        self.allOut = []
        self.allDecision = []
        return self.step(0, state)

    def writeFile(self):
        with open("data/dat" + str(self.version), "w") as f:
            f.write(str(self.__step) + "\n")
            for inp in self.allInput:
                f.write(' '.join([str(x) for x in inp]) + "\n")
            for out in self.allOut:
                f.write(' '.join([str(x) for x in out[0]]) + "\n")
            for x in self.allDecision:
                f.write(str(x) + "\n")

    def saveData(self):
        if os.path.isfile("data/dat" + str(self.version)):
            f = open("data/dat" + str(self.version), "r")
            x = int(f.readline())
            f.close()
            if x > self.__step:
                self.writeFile()
        else:
            self.writeFile()

    def step(self, reward, state):
        "Given current reward and state, agent returns next action"
        state = np.array(list(state)).reshape((1, self.__realStateDim))
        if self.__step == 0:
            self.up = self.__extractUpOrLow(state)
        if self.__step == 0:
            self.version = int(state[0, 0] * 1000)
        if reward > 9:
            for i in range(len(self.allInput)):
                x = np.array([self.allInput[i]])
                y = np.array((self.allOut[i]))
                y[0, self.allDecision[i]] = 15 - 0.01 * self.__step - 0.05 * (len(self.allInput) - i)
                # print(15 - 0.01 * self.__step - 0.05*(len(self.allInput) - i))
                self.model.fit(x, y, epochs=10, verbose=0)
            self.saveData()
        if reward > 9:
            reward *= 3
        self.__step += 1
        self.__reward += reward
        # self.__getReward(state, reward)
        orgState = state.copy()
        state = self.getFeatureVector(state, reward)
        # self.shouldAttack(state)
        l = np.array([state])
        values = self.model.predict(np.array(l))
        action = 0
        if self.__step % 4 == 1:
            if random.random() > self.e:
                action = np.argmax(values)
            else:
                action = random.randint(0, 4 * 4 - 1)
                self.strangeInput.append(state)
                self.strangeDecision.append(self.previousAction)
                self.normalOut.append(values)
        else:
            action = self.previousAction
        # if self.__step < 40:
        #     action = 0
        # elif self.__step < 95:
        #     action = 2
        # elif self.__step < 125:action=5
        # else:action=10
        if self.shouldAttack(state, orgState):
            # action = 10
            print("ATTACK")
        if self.__step > 1:
            newReward = reward + self.alpha * np.max(values)
            # if self.__step > 100:
            #     newReward = -20
            # print(newReward)
            self.previousValues[0][self.previousAction] = newReward
            self.model.fit(self.previousStep, self.previousValues, epochs=1, verbose=0)
            if self.__step > 230:
                for i in range(len(self.strangeInput)):
                    x = np.array([self.strangeInput[i]])
                    y = np.array((self.normalOut[i]))
                    y[0, self.strangeDecision[i]] = 0
                    self.model.fit(x, y, epochs=2, verbose=0)
                return "reset"

        self.previousStep = l
        self.previousValues = np.array(values)
        self.previousAction = action
        self.allDecision.append(self.previousAction)
        self.allOut.append(values)
        self.allInput.append(state)
        self.__action = self.__decodeAction(self.previousAction)
        self.e *= self.g
        # if self.__step%1001 == 0 or reward==10:
        #     print("hello")
        # Reduce chance of random action as we train the model.
        # print(self.__action)
        return self.__action

    def end(self, reward):
        # print("DONE")
        pass

    def save(self):
        if random.random() > 0.05:
            mypath = "data/"
            onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
            self.learnFromFile(mypath + onlyfiles[random.randint(0, len(onlyfiles) - 1)])
        else:
            self.learnFromAll()
        self.model.save("model44")

    def cleanup(self):
        # print("DONE1")
        pass

    def getName(self):
        return self.__name

    def __decodeAction(self, a):
        x = []
        for i in range(2):
            y = a % 4
            for j in range((5, 5)[i]):
                if self.up:
                    # może usunąć też pustą akcję? wtedy będzie tylko 3*3=9 ruchów
                    x += {0: [1, 0, 0], 1: [0, 0, 1], 2: [0, 1, 0], 3: [0, 1, 1]}[y]
                else:
                    x += {0: [0, 0, 1], 1: [1, 0, 0], 2: [0, 1, 0], 3: [1, 1, 0]}[y]
            a //= 4
        return x

    def __destruct(self):
        # print("Ala ma kota")
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        self.__destruct()

    def __del__(self):
        self.__destruct()
