#
from kmeans import kmeans
from UCI_ML_Functions import *


class test():
    data_list={'x':1,'y':2}
    

    def test_kmeans(self):
        k=1
        kmeans_exam=kmeans(k,self.data_list)
        kmeans_exam.init_center()
        kmeans_exam.kmeans_iterate()