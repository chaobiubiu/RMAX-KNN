#RMAX-KNN 和 tilecoding sarsa(lamda) 在mountaincar环境上的对比实验
import gym
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")
env=env.unwrapped

from rmax_knn_td import *
from rmax_knn_td_part import *
from rmax_knn_td_part2 import *

x=[]
y=[]

def BenchmarkExperiment(Episodes=100,k=1):
    print()
    print('- - - - - -')
    print('INIT EXPERIMENT','k='+str(k))

    Env=gym.make('MountainCar-v0')
    Env=Env.unwrapped

    IQ=IncrementKNNQ(n_max=1000,n_actions=3,n_features=2,k=k,d_target=0.09,d_point=0.16,input_ranges=[[-1.2, 0.5], [-0.07, 0.07]],RMAX=0,alpha=0.9,lr=0.95,gamma=1.0)
    As=Action_selector()
    MC=Base(IQ,Env,As,gamma=1.0)

    for i in range(Episodes):
        result=MC.SarsaEpisode(1000)
        print(len(IQ.representations))
        MC.IQ.ResetTraces()
        print('Episodes:',i,'Steps to goal:',result[1])
        if i%15==0:
            x.append(i)
            y.append(result[1])

import numpy as np

class Tilecoder:

    def __init__(self, numTilings, tilesPerTiling):
        # Set max value for normalization of inputs
        self.maxNormal = 1
        self.maxVal = env.observation_space.high
        self.minVal = env.observation_space.low
        self.numTilings = numTilings
        self.tilesPerTiling = tilesPerTiling
        self.dim = len(self.maxVal)
        self.numTiles = (self.tilesPerTiling**self.dim) * self.numTilings
        self.actions = env.action_space.n
        self.n = self.numTiles * self.actions
        self.tileSize = np.divide(np.ones(self.dim)*self.maxNormal, self.tilesPerTiling-1)

    def getFeatures(self, variables):
        # Ensures range is always between 0 and self.maxValue
        values = np.zeros(self.dim)
        #normalize the state variables
        for i in range(len(env.observation_space.shape)+1):
            values[i] = self.maxNormal * ((variables[i] - self.minVal[i])/(self.maxVal[i]-self.minVal[i]))
        tileIndices = np.zeros(self.numTilings)
        matrix = np.zeros([self.numTilings,self.dim])
        for i in range(self.numTilings):
            for i2 in range(self.dim):
                #确定归一化后的value属于当前坐标的第几个格子中，并且添加一定的偏移量
                matrix[i,i2] = int(values[i2] / self.tileSize[i2] + i / self.numTilings)
        #计算对应的格子位置，即第188个格子之类
        for i in range(1,self.dim):
            matrix[:,i] *= self.tilesPerTiling**i
        #because the second box is no more than self.tilesPerTiling**self.dim,so add the numtilings it belongs × tilesTiling××2 to the second box
        for i in range(self.numTilings):
            tileIndices[i] = (i * (self.tilesPerTiling**self.dim) + sum(matrix[i,:]))
        return tileIndices

    def oneHotVector(self, features, action):
        oneHot = np.zeros(self.n)
        for m in features:
            for n in range(self.actions):
                index_=int(m+(self.numTiles*n))
                oneHot[index_]=0
        for i in features:
            index = int(i + (self.numTiles*action))
            oneHot[index] = 1
        return oneHot

    def getVal(self, theta, features, action):
        val = 0
        for i in features:
            index = int(i + (self.numTiles*action))
            val += theta[index]
        return val

    def getQ(self, features, theta):
        Q = np.zeros(self.actions)
        for i in range(self.actions):
            Q[i] = self.getVal(theta, features, i)
        return Q

x1=[]
x2=[]
x3=[]
y1=[]
y2=[]
y3=[]

def run(numTilings,tilesPerTiling,numEpisodes,x,y):
    #tile = Tilecoder(7,14)
    tile=Tilecoder(numTilings,tilesPerTiling)
    #initial Qtable
    theta = np.random.uniform(-0.001, 0, size=(tile.n))
    alpha = (.1/ tile.numTilings)*3.2
    gamma = 1
    for episode in range(numEpisodes):
        step=0
        state = env.reset()
        while True:
            #env.render()
            F = tile.getFeatures(state)
            Q = tile.getQ(F, theta)
            action = np.argmax(Q)
            state2, reward, done, info = env.step(action)
            delta = reward - Q[action]
            if done == True or step==1000:
                #如果下一状态为终止状态，则 delta=reward+0-Q[action]
                theta += np.multiply((alpha*delta), tile.oneHotVector(F,action))

                print('episode:',episode,'total_steps:',step)
                break
            Q = tile.getQ(tile.getFeatures(state2), theta)
            delta += gamma*np.max(Q)
            theta += np.multiply((alpha*delta), tile.oneHotVector(F,action))
            state = state2
            step+=1
        if episode%15==0:
            x.append(episode)
            y.append(step)

BenchmarkExperiment(Episodes=500,k=3)
run(5,20,500,x1,y1)
run(10,20,500,x2,y2)
run(20,20,500,x3,y3)
np.savetxt('mountaincar_x.txt',x)
np.savetxt('mountaincar_y.txt',y)
np.savetxt('mountaincar_x1.txt',x1)
np.savetxt('mountaincar_y1.txt',y1)
np.savetxt('mountaincar_x2.txt',x2)
np.savetxt('mountaincar_y2.txt',y2)
np.savetxt('mountaincar_x3.txt',x3)
np.savetxt('mountaincar_y3.txt',y3)
plt.figure(num=10,figsize=(10,8))
plt.plot(x1,y1,color='red',label="Tilecoding 5",linestyle='-.',marker="x")
plt.plot(x2,y2,color='blue',label="Tilecoding 10",linestyle='--',marker="p")
plt.plot(x3,y3,color='green',label="Tilecoding 20",linestyle=':',marker=">")
#plt.plot(x,y,color='black',label="RMAX-KNN",linestyle=':')
# o x   p   s > d
plt.plot(x,y,color='black',linestyle='-',label="RMAX-KNN",marker="o")
plt.legend(loc='upper right')
plt.xlabel('Episodes')
plt.ylabel('Steps to goal')
plt.show()


