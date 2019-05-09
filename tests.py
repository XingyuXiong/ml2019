#
from kmeans import kmeans
from PCA import PCA
from UCI_ML_Functions import *
import os,sys
import numpy as np
from scipy.io import wavfile
import matplotlib.pylab as plt
from PIL import Image


START_INDEX=0
END_INDEX=3
work_path=sys.path[0]


def loadDataSet(fileName):      
    numFeat = len(open(fileName).readline().split(','))    # 计算有多少列
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():        #  遍历原始数据集每一行
        if(line=='\n'):
            continue
        lineArr =[]
        curLine = line.split(',')      # 是一列表类型,这里先采用iris数据集，以逗号分开
        for i in range(numFeat-1):     # numFeat - 1的原因：因为原始数据的最后一列是类别，不是属性数据
            lineArr.append(float(curLine[i]))  # 一个一个传进lineArr列表向量
        dataMat.append(lineArr)     # 再传进dataMat列表向量
        labelMat.append(str(curLine[-1]).strip('\n'))  # 写进标签列表，去除空格
    return dataMat, labelMat


def loadFace():
    pgm_files=[]
    path=work_path+r'/yaleB01'
    files=os.listdir(path)
    
    try:
        for f in files:
            fl=os.path.join(path,f)
            if os.path.isfile(fl) and os.path.splitext(fl)[1]=='.pgm':
                #print(fl)
                pgm_files.append(fl)
    except Exception:
        print('file search fail')
    pgm_ims=[np.array(Image.open(file)) for file in pgm_files]
    return pgm_ims



class test():
    dataMat,labelMat=loadDataSet(work_path+r'/iris/iris.data')
    data_list=dataMat

    def test_kmeans(self):
        k=3
        kmeans_exam=kmeans(k,self.data_list)
        kmeans_exam.init_center()
        kmeans_exam.iterate()


    def test_pca(self):
        pic_num=END_INDEX-START_INDEX
        images=loadFace()
        print(images[1])
        #images=[images[i][:,:]]    
        pca=PCA(images[START_INDEX:END_INDEX])
        pca_ims=pca.ret()
        for i in range(START_INDEX,END_INDEX):
            #print(pca_ims)  
            before_pca=Image.fromarray(images[i])
            #print(pca_ims[FACE_INEDX])
            after_pca=Image.fromarray(pca_ims[i-START_INDEX])
            fig=plt.figure('pca')
            ax=fig.add_subplot(pic_num,2,i+1-START_INDEX)
            ax.imshow(before_pca)
            ax=fig.add_subplot(pic_num,2,i+2-START_INDEX)
            ax.imshow(after_pca)
        plt.show()