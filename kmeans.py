#
import math


class kmeans():
    def __init__(self,knum,gauss_distrib_num=0,para_variance=0,*args):
        '''
        expect kargs as a list with multiple dimension data, each element is a dictionary, use the feature name (like x,y) as its key, the value of features as its values
        '''

        self.knum=knum
        self.data_list=kargs
        self.data_num=len(args)
        self.dim_num=len(args[0])
        self.dim_list=args[0].keys()
        self.max_data=args[0]
        self.min_data=args[0]
        self.center_list=[]
        self.center_dict={}

        self.sigma=para_variance
        self.N=gauss_distrib_num
        self.Miu=[]
        for data in args:
            for dim in data:
                self.max_data[dim]=max(self.max_data[dim],data[dim])
                self.min_data[dim]=min(self.min_data[dim],data[dim])
    
    
    def init_center(self):
        '''
        use the max and the min of single data dimension as the margin of data space, randomly choose k centers from the space
        '''

        for i in range(0,self.knum):
            self.center_list.append({key:random(self.min_data[key],self.max_data[key]) for key in self.dim_list})


    def two_norm(self,data1,data2):
        norm=0
        for dim in self.dim_list:
            norm+=(data1[dim]-data2[dim])**2
        return sqrt(norm)


    def recal_center(self):
        dim_value_sum={dim:0 for dim in self.dim_list}
        old_center_list=self.center_list
        for center in self.center_list:
            num_in_class=0
            for data in self.data_list:
                if data['class']==center.index():     
                    num_in_class+=1
                    dim_value_sum+=data #dict add

            center=dim_value_sum/num_in_class #dict div
        for i in range(0,self.knum):
            if old_center_list[i]!=self.center_list[i]: #dict equal
                return 1
        return 0


    def kmeans_iterate(self):
        for data in self.data_list:
            dis=two_norm(data,self.center_list[0])
            class_num=0
            for i in range(1,self.knum):
                if two_norm(data,self.center_list[i])<dis:
                    dis=two_norm(data,self.center_list[i])
                    class_num=i
            self.data_list['center_dis']=dis
            self.data_list['class']=class_num
        if recal_center():
            kmeans_iterate()
        return 0
            

    def GMM_iterate(self):
        '''
        use EM
        '''
        