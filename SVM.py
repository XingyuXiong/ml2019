import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd


SELECT_X=0
SELECT_Y=1
class SVM():
    def __init__(self,args,label):
        self.data=np.array(args)
        x=self.data[:,SELECT_X:SELECT_Y+1]
        y=pd.Categorical(label).codes
        x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,train_size=0.8)
        #C为惩罚因子，C越大，则会使模型更容易过拟合
        svc=svm.SVC(C=0.1,kernel='linear').fit(x_train,y_train.ravel())
        print(svc.score(x_train,y_train))
        print('Accuracy for train sets:',accuracy_score(y_train,svc.predict(x_train)))
        print(svc.score(x_test,y_test))
        print('Accuracy for test sets:',accuracy_score(y_test,svc.predict(x_test)))
