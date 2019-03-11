import numpy as np
class Base:
    def __init__(self,IQ,Env,Aselector,gamma=1.0):
        self.IQ=IQ
        self.Env = Env
        self.Aselector=Aselector
        self.gamma=gamma
        self.Aselector=Aselector
        self.Aselector.parent=self

    def SarsaEpisode(self,max_steps=1000):
        s=self.Env.reset()
        step=0
        total_reward=0
        a,v=self.Aselector(s)
        for i in range(max_steps):
            #self.Env.render()
            self.IQ.add_representations(s)
            action = np.float(a - (self.IQ.n_actions - 1) / 2) / ((self.IQ.n_actions - 1) / 4)
            s_,reward,done,info=self.Env.step(np.array([action]))
            total_reward+=reward
            a_,v_=self.Aselector(s_)
            #target的设置需要符合算法的要求，带有不确定性的rmax
            target_value=reward+self.gamma*v_*(0 if done else 1)
            self.IQ.Update(s,a,target_value)
            s=s_
            a=a_
            #考虑一下第一次到达山顶之后IQ.representations中存放着的点就是规定上一次成功路径上的状态的所有近邻点，此时是否就可以不用再exploration，即self.IQ.d_point很大，
            # 不再限制未知状态近邻点的距离限制
            if done:
                #self.IQ.d_point=0.5
                print("the episode needs %d steps"%(step))
                break
            step+=1
        return total_reward,step
