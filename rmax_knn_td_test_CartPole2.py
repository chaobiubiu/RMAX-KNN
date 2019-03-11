#rmax knn_td different d_target in cartpole
from rmax_knn_td import *
from rmax_knn_td_CartPole_part import *
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

    Env=gym.make('CartPole-v0')
    Env=Env.unwrapped

    IQ=IncrementKNNQ(n_max=10000,n_actions=2,n_features=4,k=3,d_target=d_target,d_point=0.11,input_ranges=[[-4.8, 4.8],[-3.5, 3.5],[-0.42, 0.42],[-3.5, 3.5]],
                     RMAX=1,alpha=0.2,lr=0.95,gamma=0.95)
    As=Action_selector()
    MC=Base(IQ,Env,As,gamma=0.95)

    total_step=[]
    for i in range(Episodes):
        result=MC.SarsaEpisode(200)
        print(len(IQ.representations))
        #MC.IQ.ResetTraces()
        print('Episode:',i,'Steps:',result[1])
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


BenchmarkExperiment(Episodes=200,d_target=0.03,x=x1,y=y1)
BenchmarkExperiment(Episodes=200,d_target=0.05,x=x2,y=y2)
BenchmarkExperiment(Episodes=200,d_target=0.07,x=x3,y=y3)
BenchmarkExperiment(Episodes=200,d_target=0.09,x=x4,y=y4)
BenchmarkExperiment(Episodes=200,d_target=0.11,x=x5,y=y5)
plt.figure(num=5,figsize=(10,8))
np.savetxt('cartpoledtarget_x1.txt',x1)
np.savetxt('cartpoledtarget_y1.txt',y1)
np.savetxt('cartpoledtarget_x2.txt',x2)
np.savetxt('cartpoledtarget_y2.txt',y2)
np.savetxt('cartpoledtarget_x3.txt',x3)
np.savetxt('cartpoledtarget_y3.txt',y3)
np.savetxt('cartpoledtarget_x4.txt',x4)
np.savetxt('cartpoledtarget_y4.txt',y4)
np.savetxt('cartpoledtarget_x5.txt',x5)
np.savetxt('cartpoledtarget_y5.txt',y5)
plt.plot(x1,y1,color='red',label='d_target=0.03',linestyle='-.',marker="x")
plt.plot(x2,y2,color='blue',label="d_target=0.05",linestyle=':',marker="^")
plt.plot(x3,y3,color='black',label="d_target=0.07",linestyle='-',marker="o")
plt.plot(x4,y4,color='green',label="d_target=0.09",linestyle='--',marker="s")
plt.plot(x5,y5,color='magenta',label="d_target=0.11",linestyle=':',marker="d")
plt.legend(loc='lower right')
plt.xlabel('Episodes')
plt.ylabel('Average_reward')
plt.show()
