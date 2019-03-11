import numpy as np
from scipy.spatial import cKDTree
from IncrementKNNQ_interface import *

class IncrementKNNQ(FARL):
    #n_max为离散状态空间表示点最大数目；n_actions:动作数；k为当前状态近邻点数量；d_target为新增离散状态空间表示点距离阈值；d_point有效近邻点距离阈值；
    def __init__(self,n_max,n_actions,n_features,k,d_target,d_point,input_ranges,RMAX,alpha,lr,gamma):
        self.lbounds=[]
        self.ubounds=[]
        self.n_max=n_max
        self.n_actions=n_actions
        self.n_features=n_features
        self.k=k
        self.d_target=d_target
        self.d_point=d_point
        self.RMAX=RMAX
        self.alpha=alpha
        self.lr=lr
        self.gamma=gamma
        self.Q=np.zeros((self.n_max,self.n_actions))+0.0
        self.e=np.zeros((self.n_max,self.n_actions))+0.0
        self.ac=[]
        self.m=0
        self.knn=[]
        #用来存放k个近邻之中距离<=d_point的近邻点
        self.knn_two=[]
        self.last_state=np.zeros((1,self.n_features))+0.0

        for r in input_ranges:
            self.lbounds.append(r[0])
            self.ubounds.append(r[1])
        self.lbounds=np.array(self.lbounds)
        self.ubounds=np.array(self.ubounds)
        self.representations=np.array([])

    def ResetTraces(self):
        self.e*=0.0

    def ndlinspace(self,prior_input_ranges, nelemns):
        x = np.indices(nelemns).T.reshape(-1, len(nelemns)) + 1.0
        prior_lbounds = []
        prior_ubounds = []
        from_b = np.array(nelemns, float)
        for r in prior_input_ranges:
            prior_lbounds.append(r[0])
            prior_ubounds.append(r[1])
        prior_lbounds = np.array(prior_lbounds, float)
        prior_ubounds = np.array(prior_ubounds, float)
        y = (prior_lbounds) + (((x - 1) / (from_b - 1)) * ((prior_ubounds) - (prior_lbounds)))
        return y

    #归一化函数
    def ScaleValue(self,x,from_a,from_b,to_a,to_b):
        return (to_a)+((x-from_a)/(from_b-from_a))*((to_b)-(to_a))

    def RescaleInputs(self,s):
        return self.ScaleValue(np.array(s),self.lbounds,self.ubounds,-1.0,1.0)

    #GetKNNSet函数返回当前状态s的k个近邻点中与其距离<=d_point的近邻点集合knn_two以及无效近邻点的个数m
    def GetKNNSet(self,s):
        self.m=0
        self.store=[]
        self.d_store=[]
        self.uncertain=[]
        self.uncertain_scale=0
        if np.allclose(s,self.last_state) and self.knn!=[]:
            return self.knn
        self.last_state=s
        state=self.RescaleInputs(s)
        #根据目前已有的状态空间表示点集合self.representations中的点来构造cKDTree
        #cKDTree返回当前状态k个近邻点的集合self.knn以及距离集合self.d
        if len(self.representations)!=0:
            cl=np.array(self.representations)
            self.knntree=cKDTree(cl,100)
            self.d,self.knn=self.knntree.query(state,self.k,eps=0.0,p=2)
            #判断近邻点中与当前状态距离<=阈值d_point的近邻点集合self.store
            #距离集合self.d_store 以及符合要求的近邻点个数m
            for i in range(self.k):
                if self.k==1:
                    if self.d<=self.d_point:
                        self.d_store.append(self.d)
                        self.store.append(self.knn)
                    else:
                        #m用来记录无效近邻点的个数
                        self.m += 1
                else:
                    if self.d[i]<=self.d_point:
                        self.d_store.append(self.d[i])
                        self.store.append(self.knn[i])
                    else:
                        #m用来记录无效近邻点的个数
                        self.m += 1
            self.knn_two=self.store
            self.d=np.array(self.d)
            self.d*=self.d
            self.ac=1.0/(1.0+self.d)
            if self.k==1:
                self.ac/=self.ac
            else:
                self.ac/=sum(self.ac)
            return self.knn_two
        else:
            self.m=self.k
            return []

    def CalQValues(self,M):
        #如果M==[]，则当前状态所有的Q(s,a)皆设置为Q_MAX；否则将其无效近邻点对应的状态行为值函数设为Q_MAX，有效近邻点对应的状态行为值函数仍为Q表中存储的值。
        if M==[]:
            return np.ones((self.k,self.n_actions))*(0)
        else:
            number=self.m
            Q_uncertain=np.ones((number,self.n_actions))*(0)
            QValues=np.dot(np.transpose(np.vstack((self.Q[M],Q_uncertain))),self.ac)
            return QValues

    def GetValue(self,s,a=None):
        M=self.GetKNNSet(s)
        if a==None:
            return self.CalQValues(M)
        return self.CalQValues(M)[a]

    def Update(self,s,a,v,gamma=1.0):
        M=self.GetKNNSet(s)
        L=self.knn
        if self.lr>0:
            #replacing traces
            self.e[L]=0
            self.e[L,a]=self.ac
            TD_error=v-self.GetValue(s,a)
            self.Q+=self.alpha*(TD_error)*self.e
            self.e*=self.lr
        else:
            TD_error=v-self.GetValue(s,a)
            self.Q[L,a]+=self.alpha*(TD_error)*self.ac

    #还得编写一个添加离散状态空间代表点的方法
    def add_representations(self,s):
        if len(self.representations)==0:
            self.representations=np.array(self.RescaleInputs(s).reshape(1, -1))
        else:
            min_distance=np.linalg.norm(self.RescaleInputs(s)-self.representations[0])
            for i in range(len(self.representations)):
                distance=np.linalg.norm(self.RescaleInputs(s)-self.representations[i])
                if distance<=min_distance:
                    min_distance=distance
            if min_distance>=self.d_target and len(self.representations)<self.n_max:
                self.representations = np.vstack((self.representations, self.RescaleInputs(s).reshape(1, -1)))
        return self.representations












