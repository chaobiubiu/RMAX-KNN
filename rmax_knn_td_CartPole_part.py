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
            s_,reward,done,info=self.Env.step(a)
            if done:
                reward=-max_steps
            total_reward+=reward
            a_,v_=self.Aselector(s_)
            target_value=reward+self.gamma*v_*(0 if done else 1)
            self.IQ.Update(s,a,target_value)
            s=s_
            a=a_
            if done:
                print("the episode needs %d steps"%(step))
                break
            step+=1
        return total_reward,step
