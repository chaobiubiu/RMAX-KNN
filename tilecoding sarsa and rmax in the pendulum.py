#RMAX-KNN 和tilecoding sarsa(lamda)在pendulum环境下的对比试验

from rmax_knn_td import *
from rmax_knn_td_pendulum_part import *
from rmax_knn_td_part2 import *
import gym
import matplotlib.pyplot as plt

x=[]
y=[]

def BenchmarkExperiment(Episodes=100,k=1):
    print()
    print('- - - - - -')
    print('INIT EXPERIMENT','k='+str(k))

    Env=gym.make('Pendulum-v0')
    Env=Env.unwrapped

    IQ=IncrementKNNQ(n_max=10000,n_actions=5,n_features=3,k=k,d_target=0.06,d_point=0.08,input_ranges=[[-1,1],[-1,1],[-8,8]],RMAX=0,alpha=0.9,lr=0.95,gamma=0.95)
    As=Action_selector()
    MC=Base(IQ,Env,As,gamma=0.95)

    for i in range(Episodes):
        result=MC.SarsaEpisode(500)
        print(len(IQ.representations))
        #MC.IQ.ResetTraces()
        print('Episodes:',i,'Total_reward:',result[0])
        if i %15==0:
            x.append(i)
            y.append(result[0])



import numpy as np

env = gym.make("Pendulum-v0")
import random as rand


class tilecoder:

    def __init__(self, numTilings, tilesPerTiling):
        self.maxIn = env.observation_space.high
        self.minIn = env.observation_space.low
        self.numTilings = numTilings
        self.tilesPerTiling = tilesPerTiling
        self.dim = len(self.maxIn)
        self.numTiles = (self.tilesPerTiling ** self.dim) * self.numTilings
        self.actions = 11
        self.n = self.numTiles * self.actions
        self.tileSize = np.divide(np.subtract(self.maxIn, self.minIn), self.tilesPerTiling - 1)

    def getFeatures(self, variables):
        ### ENSURES LOWEST POSSIBLE INPUT IS ALWAYS 0
        self.variables = np.subtract(variables, self.minIn)
        tileIndices = np.zeros(self.numTilings)
        matrix = np.zeros([self.numTilings, self.dim])
        for i in range(self.numTilings):
            for i2 in range(self.dim):
                matrix[i, i2] = int(self.variables[i2] / self.tileSize[i2] \
                                    + i / self.numTilings)
        for i in range(1, self.dim):
            matrix[:, i] *= self.tilesPerTiling ** i
        for i in range(self.numTilings):
            tileIndices[i] = (i * (self.tilesPerTiling ** self.dim) \
                              + sum(matrix[i, :]))
        return tileIndices

    def oneHotVector(self, features, action):
        oneHot = np.zeros(self.n)
        for m in features:
            for n in range(self.actions):
                index_=int(m+(self.numTiles*n))
                oneHot[index_]=0

        for i in features:
            index = int(i + (self.numTiles * action))
            oneHot[index] = 1
        return oneHot

    def getVal(self, theta, features, action):
        val = 0
        for i in features:
            index = int(i + (self.numTiles * action))
            val += theta[index]
        return val

    def getQ(self, features, theta):
        Q = np.zeros(self.actions)
        for i in range(self.actions):
            Q[i] = self.getVal(theta, features, i)
        return Q

    def action2float(self, a):
        if a == 0:
            return ([rand.uniform(-2, -1.5)])
        if a == 1:
            return ([rand.uniform(-1.5, -1)])
        if a == 2:
            return ([rand.uniform(-1, -0.75)])
        if a == 3:
            return ([rand.uniform(-0.75, -0.5)])
        if a == 4:
            return ([rand.uniform(-0.5, -0.25)])
        if a == 5:
            return ([0])
        if a == 6:
            return ([rand.uniform(0.25, 0.5)])
        if a == 7:
            return ([rand.uniform(0.5, 0.75)])
        if a == 8:
            return ([rand.uniform(0.75, 1)])
        if a == 9:
            return ([rand.uniform(1, 1.5)])
        if a == 10:
            return ([rand.uniform(1.5, 2)])

x1=[]
y1=[]
x2=[]
y2=[]
x3=[]
y3=[]

#4,20
def run(numTilings,tilesPerTiling,numEpisodes,x,y):

    tile = tilecoder(numTilings, tilesPerTiling)
    theta = np.random.uniform(-0.001, 0, size=(tile.n))
    alpha = .1 / tile.numTilings*2
    gamma=0.95

    for episodeNum in range(numEpisodes):
        G = 0
        state = env.reset()
        steps=0
        while True:
            #env.render()
            F = tile.getFeatures(state)
            Q = tile.getQ(F, theta)
            action = np.argmax(Q)

            state2, reward, done, info = env.step(tile.action2float(action))
            G += reward

            Q_ = tile.getQ(tile.getFeatures(state2), theta)
            delta = reward + gamma *np.max(Q_) - Q[action]
            theta += np.multiply((alpha * delta), tile.oneHotVector(F, action))
            if steps==499:
                print("epsiode:",episodeNum,"total_reward:",G)
                break


            state = state2
            steps+=1

        if episodeNum%15==0:
            x.append(episodeNum)
            y.append(G)


BenchmarkExperiment(Episodes=500,k=3)
run(5,20,500,x1,y1)
run(10,20,500,x2,y2)
run(20,20,500,x3,y3)
np.savetxt('pendulum_x.txt',x)
np.savetxt('pendulum_y.txt',y)
np.savetxt('pendulum_x1.txt',x1)
np.savetxt('pendulum_y1.txt',y1)
np.savetxt('pendulum_x2.txt',x2)
np.savetxt('pendulum_y2.txt',y2)
np.savetxt('pendulum_x3.txt',x3)
np.savetxt('pendulum_y3.txt',y3)
plt.figure(num=12,figsize=(10,8))
plt.plot(x1,y1,color='red',label="Tilecoding 5",linestyle='-.',marker="x")
plt.plot(x2,y2,color='blue',label="Tilecoding 10",linestyle='--',marker="p")
plt.plot(x3,y3,color='green',label="Tilecoding 20",linestyle=':',marker=">")
#plt.plot(x,y,color='black',label="RMAX-KNN",linestyle=':')
# o x   p   s > d
plt.plot(x,y,color='black',linestyle='-',label="RMAX-KNN",marker="o")
plt.xlabel('Episodes')
plt.ylabel('Total_reward')
plt.legend(loc='lower right')
plt.show()

