#
import math
from random import uniform
import matplotlib.pyplot as plt
import numpy as np
DIM_X=0
DIM_Y=1
MAX_ITER_DEPTH=3
ITER_LIMIT=False
DRAW=False

class kmeans():
    def __init__(self,knum,args):
        '''
        expect args as a list with multiple dimension data, each element is a dictionary, use the feature name (like x,y) as its key, the value of features as its values
        '''

        self.knum=knum
        self.data=np.array(args)#二维数组，数据矩阵
        self.d_len=len(self.data[0])#数据特征长度
        self.max_data=args[0][:]#必须要加.copy或者[:](仅限于list)否则max_data min_data共享内存地址,因为此处不用改变args了所以[:]复制也可以
        self.min_data=args[0][:]
        self.center_list=[]#kmeans递归中生成的聚类中心
        
        self.data_class=[0 for i in range(len(self.data))]#该数组用于指示当前循环中某个数据点属于某类，下标顺序与self.data相同
        #self.sigma=para_variance
        #self.N=gauss_distrib_num
        #self.Miu=[]
        self.max_data=self.data.max(axis=0)
        self.min_data=self.data.min(axis=0)
    
        self.colors=np.array([[i//4%2,i//2%2,i%2] for i in range(1,self.knum+1)])#rgb三色组
        self.iter_depth=0#当前函数执行次数（递归深度）
        self.fig, self.axs = plt.subplots(MAX_ITER_DEPTH+1,2, figsize=(10, 10))

    def init_center(self):
        '''
        use the max and the min of single data dimension as the margin of data space, randomly choose k centers from the space
        '''
        for i in range(0,self.knum):
            center=uniform(self.min_data,self.max_data)
            self.center_list.append(center)
        self.center_list=np.array(self.center_list)


    def iterate(self):
        self.kmeans_iterate()#本来想用do while的，python好像没有哈哈
        while(self.recal_center()):
            self.kmeans_iterate()
        self.print(self.axs,MAX_ITER_DEPTH)#输出聚类收敛的结果
        plt.show()


    def two_norm(self,data1,data2):
        norm=0
        for i in range(self.d_len):
            norm+=(data1[i]-data2[i])**2
        return norm**0.5


    def recal_center(self):
        old_center_list=self.center_list.copy()#不能用[:]而用浅复制,这两者有区别
        for i in range(self.knum):
            '''
            以下相当于计算某类中所有点的平均几何中心，本来想用np中的数组sum运算再除一下，但self.data_class这个东西我试了下用化为np矩阵算有一些问题，
            np矩阵不太好进行大规模增改，同时指定元素获取下标也不是太简单，还是用普通的一维数组一步一步写吧
            本来的思路是np矩阵的每一个列向量代表某类点的集合，这样axis=0统计和非常方便，但是要在一个矩阵中查找指定元素把它挪到另一个列向量里就很麻烦了
            '''
            sum=np.zeros(self.d_len)
            num=0
            for j in range(len(self.data)):    
                if i==self.data_class[j]:
                    num+=1
                    sum+=self.data[j]
            if num==0:
                continue
            self.center_list[i]=sum/num
        if (old_center_list==self.center_list).all():#检验结果是否已经收敛
            print('stop iterate')
            return 0
        self.iter_depth+=1
        return 1


    def kmeans_iterate(self):       
        if self.iter_depth>=MAX_ITER_DEPTH and ITER_LIMIT:#用于提前终止递归或者避免永远递归
            return 0
        for i in range(len(self.data)):
            data=self.data[i]
            dis=np.array([self.two_norm(data,self.center_list[i]) for i in range(self.knum)])
            class_num=np.argmin(dis)#获取dis数组中最小值的下标，即找到离该数据点最近的聚类中心（二范数就是欧式距离）
            self.data_class[i]=class_num
        if self.iter_depth<MAX_ITER_DEPTH:#查看前几次递归的结果
            self.print(self.axs,self.iter_depth)
        return 0
            

    def GMM_iterate(self):
        '''
        use EM
        '''
        
    def print(self,axs,pic_n):
        selectx=DIM_X
        selecty=DIM_Y
        x=self.data[:,selectx]
        y=self.data[:,selecty]#显然，对于iris的四维数据空间，只能选取其中两项或三项来展示，这也导致聚类中心“看起来”不一定在中间
        c=[self.colors[i] for i in self.data_class]
        axs[pic_n,0].scatter(x=x,y=y,c=c,s=10)
        axs[pic_n,0].scatter(x=self.center_list[:,selectx],y=self.center_list[:,selecty],c=self.colors,marker='o',s=50,edgecolors='k')
        #以下两项可以除去，仅适用于当前数据集（iris）
        axs[pic_n,1].scatter(x=self.data[:,2],y=self.data[:,3],c=c,s=10)
        axs[pic_n,1].scatter(x=self.center_list[:,2],y=self.center_list[:,3],c=self.colors,marker='o',s=50,edgecolors='k')