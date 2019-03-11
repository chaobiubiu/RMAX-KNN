#RMAX-KNN and tilecoding sarsa(lamda) in the cartpole
import numpy as np
import gym

from rmax_knn_td import *
from rmax_knn_td_CartPole_part import *
from rmax_knn_td_part2 import *
import matplotlib.pyplot as plt

x=[]
y=[]

def BenchmarkExperiment(Episodes=100,k=1):
    print()
    print('- - - - - -')
    print('INIT EXPERIMENT','k='+str(k))

    Env=gym.make('CartPole-v0')
    Env=Env.unwrapped

    IQ=IncrementKNNQ(n_max=10000,n_actions=2,n_features=4,k=k,d_target=0.07, d_point=0.11,
                       input_ranges=[[-4.8, 4.8], [-3.5, 3.5], [-0.42, 0.42], [-3.5, 3.5]],RMAX=1,alpha=0.2,lr=0.95,gamma=0.95)
    As=Action_selector()
    MC=Base(IQ,Env,As,gamma=0.95)

    total_step=[]
    for i in range(Episodes):
        result=MC.SarsaEpisode(200)
        print(len(IQ.representations))
        #MC.IQ.ResetTraces()
        print('Episodes:',i,'Total_reward:',result[1])
        total_step.append(result[1])
        if i>=99:
            print(np.mean(total_step[-100:]))
        if (i+1)<100:
            if i%15==0:
                x.append(i)
                y.append(np.mean(total_step))
        else:
            if i%15==0:
                x.append(i)
                y.append(np.mean(total_step[-100:]))


env = gym.make("CartPole-v0")

#tilecoding sarsa代码是通过参考https://github.com/wagonhelm/Tilecoder实现，但是源码有时会报错，因此申请空间时空间大小设为1.5倍，这只是为了防止代码报错，并不影响算法原本的复杂度。
class tilecoder:

    def __init__(self, numTilings, tilesPerTiling):
        #self.maxIn = [3, 3.5, 0.25, 3.5]
        self.maxIn = [3, 3.5, 0.25, 3.5]
        # self.minIn = env.observation_space.low
        #self.minIn = [-3, -3.5, -0.25, -3.5]
        self.minIn = [-3, -3.5, -0.25, -3.5]
        self.numTilings = numTilings
        self.tilesPerTiling = tilesPerTiling
        self.dim = len(self.maxIn)
        self.numTiles = (self.tilesPerTiling ** self.dim) * self.numTilings
        self.actions = env.action_space.n
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
        oneHot = np.zeros(1.5*self.n)
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

x1=[]
y1=[]
x2=[]
y2=[]
x3=[]
y3=[]
#4 22
def run(numTilings,tilesPerTiling,numEpisodes,x,y):
    tile = tilecoder(numTilings, tilesPerTiling)
    #theta = np.random.uniform(-0.001, 0, size=(tile.n))
    theta = np.random.uniform(-0.001, 0, size=1.5*(tile.n))
    alpha = (0.1/ tile.numTilings)
    gamma = 1
    epsilon = 1.0
    total_step=[]

    for episode in range(numEpisodes):
        step = 0
        G=0
        state = env.reset()
        while True:
            #env.render()
            F = tile.getFeatures(state)
            Q = tile.getQ(F, theta)

            if np.random.uniform() > epsilon:
                action = env.action_space.sample()
                epsilon += epsilon * 0.00005
            else:
                action = np.argmax(Q)

            state2, reward, done, info = env.step(action)
            '''if done:
                reward=-200'''
            G += reward
            delta = G * (gamma ** step) - Q[action]
            #delta=reward-Q[action]

            #delta = reward - Q[action]

            if done == True or step==200:
                theta += np.multiply((alpha * delta), tile.oneHotVector(F, action))
                print('episode:',episode,'total_steps:',step)
                break
            Q = tile.getQ(tile.getFeatures(state2), theta)
            delta += gamma * np.max(Q)
            theta += np.multiply((alpha * delta), tile.oneHotVector(F, action))
            state = state2
            step+=1
        total_step.append(step)
        if episode>=99:
            print(np.mean(total_step[-100:]))
        if (episode+1)<100:
            if episode%15==0:
                x.append(episode)
                y.append(np.mean(total_step))
        else:
            if episode%15==0:
                x.append(episode)
                y.append(np.mean(total_step[-100:]))



run(5,20,500,x1,y1)
np.savetxt('newcartpole_x1.txt',x1)
np.savetxt('newcartpole_y1.txt',y1)
run(10,20,500,x2,y2)
np.savetxt('newcartpole_x2.txt',x2)
np.savetxt('newcartpole_y2.txt',y2)
run(20,20,500,x3,y3)
np.savetxt('newcartpole_x3.txt',x3)
np.savetxt('newcartpole_y3.txt',y3)
BenchmarkExperiment(Episodes=500,k=3)
np.savetxt('newcartpole_x.txt',x)
np.savetxt('newcartpole_y.txt',y)
print('the average steps of algorithm 1 is ',sum(y1)/len(x1))
print('the average steps of algorithm 2 is ',sum(y2)/len(x2))
print('the average steps of algorithm 3 is ',sum(y3)/len(x3))
print('the average steps of my algorithm is ',sum(y)/len(x))
plt.figure(num=11,figsize=(10,8))
plt.plot(x1,y1,color='red',label="Tilecoding 5 Sarsa(λ)",linestyle='-.',marker="x")
plt.plot(x2,y2,color='blue',label="Tilecoding 10 Sarsa(λ)",linestyle='--',marker="s")
plt.plot(x3,y3,color='green',label="Tilecoding 20 Sarsa(λ)",linestyle=':',marker=">")
#plt.plot(x,y,color='black',label="RMAX-KNN",linestyle=':')
# o x   p   s > d
plt.plot(x,y,color='black',linestyle='-',label="RMAX-KNN",marker="o")
plt.legend(loc='lower right')
plt.xlabel('Episodes')
plt.ylabel('Average_reward')
plt.show()