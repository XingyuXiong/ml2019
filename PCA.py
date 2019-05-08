import numpy as np

REMAIN_RATE=0.95

class PCA():
    def __init__(self,args):
        self.raw_data=np.array(args)
        self.data=self.centerize(self.raw_data)
        self.pca(self.data)


    def centerize(self,args):
        return args-args.sum()/args.shape[0]


    def pca(self,X):
        Xt=X.transpose()
        self.X=X
        self.Xt=Xt
        m=X.shape[0]
        self.cov=np.matmul(Xt,X)/(m-1)
        self.lamda_v,self.lamda_a=np.linalg.eig(self.cov)
        lamda={self.lamda_v[i]:self.lamda_a[i] for i in range(len(self.lamda_a))}
        lamda=sorted(lamda.items(),key=lambda x:x[0],reverse=True)
        print(lamda)
        sum=np.sum(self.lamda_v)*REMAIN_RATE
        remain_sum=0
        self.pca_mat=[]
        for i in range(len(lamda)):
            if(remain_sum<REMAIN_RATE):
                remain_sum+=lamda[i][0]
                self.pca_mat.append(lamda[i][1])
        print(self.pca_mat)


    def svd(self):
        self.U=np.matmul(self.X,self.Xt)
        self.V=np.matmul(self.Xt,self.X)
        self.S=self.lamda_v**0.5