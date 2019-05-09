import numpy as np
from sklearn import svm


class SVM(svm):
    def __init__(self,args):
        self.data=np.array(args)
        #C为惩罚因子，C越大，则会使模型更容易过拟合
        svc=svm.SVC(C=0.1,kernel='linear').fit(self.data[:,0],self.data[:,1].reshape(-1))