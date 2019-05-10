import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt


SELECT_X=0
SELECT_Y=1
class SVM():
    def __init__(self,args,label,kernel='linear'):
        self.data=np.array(args)
        x=self.data[:,SELECT_X:SELECT_Y+1]
        y=pd.Categorical(label).codes
        x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,train_size=0.8)
        #C为惩罚因子，C越大，则会使模型更容易过拟合
        svc=svm.SVC(C=0.1,kernel=kernel).fit(x_train,y_train.ravel())
        y_train_pre=svc.predict(x_train)
        y_test_pre=svc.predict(x_test)
        print('Accuracy for train sets:',accuracy_score(y_train,y_train_pre))
        print('Accuracy for test sets:',accuracy_score(y_test,y_test_pre))

        fig=plt.figure('')
        ax=fig.add_subplot(311)
        ax.scatter(x[:,0],x[:,1],c=y)
        ax=fig.add_subplot(312)
        ax.scatter(x_train[:,0],x_train[:,1],c=y_train_pre)
        ax=fig.add_subplot(313)
        ax.scatter(x_test[:,0],x_test[:,1],c=y_test_pre)
        plt.show()