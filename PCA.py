import numpy as np

REMAIN_DIM=80
REMAIN_RATE=0.99
ENABLE_RATE=False

class PCA():
    def __init__(self,args):
        self.raw_data=np.array(args)
        self.len=len(self.raw_data)
        self.data=self.centerize(self.raw_data)
        print(self.data)


    def centerize(self,args):
        center_data=[0 for i in range(self.len)]
        for i in range(self.len):
            center_data[i]=np.array([arr-(arr.sum()/len(arr)) for arr in args[i]])
        return center_data


    def pca(self,X):
        
        m=X.shape[0] 
        self.cov=np.matmul(X,X.transpose())/(m-1)
        for i in self.cov:
            for j in i:
                j=round(j,2)
        print(self.cov)
        self.lamda_v,self.lamda_a=np.linalg.eig(self.cov)
        lamda={self.lamda_v[i]:self.lamda_a[i] for i in range(len(self.lamda_a))}
        lamda=sorted(lamda.items(),key=lambda x:x[0],reverse=True)
        print(self.lamda_v)

        sum=np.sum(self.lamda_v)*REMAIN_RATE
        remain_sum=0
        self.pca_mat=[]
        if ENABLE_RATE:
            for i in range(len(lamda)):
                if(remain_sum<sum):
                    remain_sum+=lamda[i][0]
                    self.pca_mat.append(lamda[i][1])
        else:
            for i in range(len(lamda)):
                if i<REMAIN_DIM:
                    self.pca_mat.append(lamda[i][1])
        self.pca_mat=np.array(self.pca_mat)
        print(self.pca_mat)
        #print(np.matmul(self.pca_mat.transpose(),self.pca_mat).shape)
        #print(self.cov)
        return self.pca_mat


    def ret(self):
        self.return_data=[0 for i in range(self.len)]
        for i in range(len(self.data)):
            tran_mat=self.pca(self.data[i])
            tran_mat=np.matmul(tran_mat.transpose(),tran_mat)
            rev_pca_data=np.matmul(tran_mat,self.data[i])
            print(rev_pca_data.shape)
            self.return_data[i]=rev_pca_data+self.raw_data[i]-self.data[i]
        return self.return_data


    def svd(self):
        self.U=np.matmul(self.X,self.Xt)
        self.V=np.matmul(self.Xt,self.X)
        self.S=self.lamda_v**0.5