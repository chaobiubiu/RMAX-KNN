#different d_target in MountainCar
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

def BenchmarkExperiment(Episodes,d_target,x,y):
    print()
    print('- - - - - -')
    print('INIT EXPERIMENT','d_target='+str(d_target))

    Env=gym.make('MountainCar-v0')
    Env=Env.unwrapped

    IQ=IncrementKNNQ(n_max=5000,n_actions=3,n_features=2,k=3,d_target=d_target,d_point=0.16,\
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

BenchmarkExperiment(Episodes=100,d_target=0.05,x=x1,y=y1)
BenchmarkExperiment(Episodes=100,d_target=0.07,x=x2,y=y2)
BenchmarkExperiment(Episodes=100,d_target=0.09,x=x3,y=y3)
BenchmarkExperiment(Episodes=100,d_target=0.11,x=x4,y=y4)
BenchmarkExperiment(Episodes=100,d_target=0.13,x=x5,y=y5)
plt.figure(num=2,figsize=(10,8))
np.savetxt('mountaincardtarget_x1.txt',x1)
np.savetxt('mountaincardtarget_y1.txt',y1)
np.savetxt('mountaincardtarget_x2.txt',x2)
np.savetxt('mountaincardtarget_y2.txt',y2)
np.savetxt('mountaincardtarget_x3.txt',x3)
np.savetxt('mountaincardtarget_y3.txt',y3)
np.savetxt('mountaincardtarget_x4.txt',x4)
np.savetxt('mountaincardtarget_y4.txt',y4)
np.savetxt('mountaincardtarget_x5.txt',x5)
np.savetxt('mountaincardtarget_y5.txt',y5)
plt.plot(x1,y1,color='red',label="d_target=0.05",linestyle=':',marker="x")
plt.plot(x2,y2,color='blue',label="d_target=0.07",linestyle='--',marker="^")
plt.plot(x3,y3,color='black',label="d_target=0.09",linestyle='-',marker="o")
plt.plot(x4,y4,color='green',label="d_target=0.11",linestyle=':',marker="s")
plt.plot(x5,y5,color='magenta',label="d_target=0.13",linestyle='-.',marker="d")
plt.legend(loc='upper right')
plt.xlabel('Episodes')
plt.ylabel('Steps to goal')
plt.show()
