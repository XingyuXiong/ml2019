#
from kmeans import kmeans


class test():
    data_list=[{'x':1}]
    

    def test_kmeans(self):
        k=0
        kmeans_exam=kmeans(k,self.data_list)
        kmeans_exam.iterate()
        
