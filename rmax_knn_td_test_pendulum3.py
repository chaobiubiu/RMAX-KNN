#different d_point in pendulum
from rmax_knn_td import *
from rmax_knn_td_pendulum_part import *
from rmax_knn_td_part2 import *
import gym
import matplotlib.pyplot as plt

x1=[]
y1=[]
x2=[]
y2=[]
x3=[]
y3=[]
x4=[]
y4=[]
x5=[]
y5=[]

def BenchmarkExperiment(Episodes,d_point,x,y):
    print()
    print('- - - - - -')
    print('INIT EXPERIMENT','d_point='+str(d_point))

    Env=gym.make('Pendulum-v0')
    Env=Env.unwrapped

    IQ=IncrementKNNQ(n_max=10000,n_actions=5,n_features=3,k=3,d_target=0.05,d_point=d_point,input_ranges=[[-1,1],[-1,1],[-8,8]],RMAX=0,alpha=0.9,lr=0.95,gamma=0.95)
    As=Action_selector()
    MC=Base(IQ,Env,As,gamma=0.95)

    for i in range(Episodes):
        result=MC.SarsaEpisode(500)
        print(len(IQ.representations))
        #MC.IQ.ResetTraces()
        print('Episodes:',i,'Total_reward:',result[0])
        if i %12==0:
            x.append(i)
            y.append(result[0])

BenchmarkExperiment(Episodes=100,d_point=0.06,x=x1,y=y1)
BenchmarkExperiment(Episodes=100,d_point=0.07,x=x2,y=y2)
BenchmarkExperiment(Episodes=100,d_point=0.09,x=x3,y=y3)
BenchmarkExperiment(Episodes=100,d_point=0.11,x=x4,y=y4)
BenchmarkExperiment(Episodes=100,d_point=0.13,x=x5,y=y5)
plt.figure(num=9,figsize=(10,8))
np.savetxt('pendulumdpoint_x1.txt',x1)
np.savetxt('pendulumdpoint_y1.txt',y1)
np.savetxt('pendulumdpoint_x2.txt',x2)
np.savetxt('pendulumdpoint_y2.txt',y2)
np.savetxt('pendulumdpoint_x3.txt',x3)
np.savetxt('pendulumdpoint_y3.txt',y3)
np.savetxt('pendulumdpoint_x4.txt',x4)
np.savetxt('pendulumdpoint_y4.txt',y4)
np.savetxt('pendulumdpoint_x5.txt',x5)
np.savetxt('pendulumdpoint_y5.txt',y5)
plt.plot(x1,y1,color='red',label='d_point=0.06',linestyle='-.',marker="x")
plt.plot(x2,y2,color='blue',label="d_point=0.07",linestyle='--',marker="^")
plt.plot(x3,y3,color='black',label="d_point=0.09",linestyle='-',marker="o")
plt.plot(x4,y4,color='green',label="d_point=0.11",linestyle=':',marker="s")
plt.plot(x5,y5,color='magenta',label="d_point=0.13",linestyle='-.',marker="d")
plt.legend(loc='lower right')
plt.xlabel('Episodes')
plt.ylabel('Total_reward')
plt.show()
