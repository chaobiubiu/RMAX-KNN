#different d_point in MountainCar
from rmax_knn_td import *
from rmax_knn_td_part import *
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

    Env=gym.make('MountainCar-v0')
    Env=Env.unwrapped

    IQ=IncrementKNNQ(n_max=1000,n_actions=3,n_features=2,k=3,d_target=0.09,d_point=d_point,\
                     input_ranges=[[-1.2, 0.5], [-0.07, 0.07]],RMAX=0,alpha=0.9,lr=0.95,gamma=1.0)
    As=Action_selector()
    MC=Base(IQ,Env,As,gamma=1.0)

    for i in range(Episodes):
        result=MC.SarsaEpisode(1000)
        print(len(IQ.representations))
        MC.IQ.ResetTraces()
        print('Episode:',i,'Steps:',result[1])
        if i%12==0:
            x.append(i)
            y.append(result[1])

BenchmarkExperiment(Episodes=100,d_point=0.10,x=x1,y=y1)
BenchmarkExperiment(Episodes=100,d_point=0.12,x=x2,y=y2)
BenchmarkExperiment(Episodes=100,d_point=0.14,x=x3,y=y3)
BenchmarkExperiment(Episodes=100,d_point=0.16,x=x4,y=y4)
BenchmarkExperiment(Episodes=100,d_point=0.18,x=x5,y=y5)
plt.figure(num=3,figsize=(10,8))
np.savetxt('mountaincardpoint_x1.txt',x1)
np.savetxt('mountaincardpoint_y1.txt',y1)
np.savetxt('mountaincardpoint_x2.txt',x2)
np.savetxt('mountaincardpoint_y2.txt',y2)
np.savetxt('mountaincardpoint_x3.txt',x3)
np.savetxt('mountaincardpoint_y3.txt',y3)
np.savetxt('mountaincardpoint_x4.txt',x4)
np.savetxt('mountaincardpoint_y4.txt',y4)
np.savetxt('mountaincardpoint_x5.txt',x5)
np.savetxt('mountaincardpoint_y5.txt',y5)
plt.plot(x1,y1,color='red',label='d_point=0.10',linestyle=':',marker="x")
plt.plot(x2,y2,color='blue',label="d_point=0.12",linestyle='--',marker="^")
plt.plot(x3,y3,color='green',label="d_point=0.14",linestyle=':',marker="s")
plt.plot(x4,y4,color='black',label="d_point=0.16",linestyle='-',marker="o")
plt.plot(x5,y5,color='magenta',label="d_point=0.18",linestyle='-.',marker="d")
plt.legend(loc='upper right')
plt.xlabel('Episodes')
plt.ylabel('Steps to goal')
plt.show()
