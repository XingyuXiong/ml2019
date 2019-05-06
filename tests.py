#
from kmeans import kmeans
from UCI_ML_Functions import *
import os,sys

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


class test():
    dataMat,labelMat=loadDataSet(sys.path[0]+r'/iris/iris.data')
    data_list=dataMat

    def test_kmeans(self):
        #print(sys.path[0])
        k=3
        kmeans_exam=kmeans(k,self.data_list)
        kmeans_exam.init_center()
        kmeans_exam.iterate()