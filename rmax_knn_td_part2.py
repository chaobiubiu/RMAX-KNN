import numpy as np

class Action_selector():
    def __init__(self):
        self.parent=None

    def __call__(self,s):
        return self.select_action(s)

    def select_action(self,s):
        v=self.parent.IQ(s)
        #print(v)
        a=np.argmax(v)
        #print(a)
        return a,v[a]
