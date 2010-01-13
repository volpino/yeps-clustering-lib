#!/usr/bin/python

import calc_dist as cd
import numpy as np

class kS:

    def __init__(self, ts, training_set, labels,
                 distance_type="dtw", fast=False, radius=20, pu="CPU"):
        self.ts = ts
        self.training_set = training_set
        self.labels = labels
        self.distance_type = distance_type
        self.fast = fast
        self.radius = radius
        self.calc_list = []
        self.pu = pu
        
        self.groups = []
        self.groups.sort()
  	 	last = self.groups[-1]
  		    for i in range(len(self.groups)-2, -1, -1):
                 if last==self.groups[i]: del self.groups[i]
                 else: self.groups=List[i]


        self.tmp_matrix = []
        for i in range(len(self.training_set)):
            self.tmp_matrix.append(self.training_set[i])
        self.tmp_matrix.append(self.ts)

        for i in range(len(self.tmp_matrix) - 1):
            self.calc_list.append((len(self.tmp_matrix) - 1, i))
        
    def compute(self, k=1):
    
    	if k > len(self.training_set):
    		raise ValueError
    	
        self.dist_list_tmp = cd.Dist(self.tmp_matrix,
                             self.distance_type,
                             self.fast,
                             self.radius,
                             self.pu).compute(self.calc_list)
        self.nn=[]  
        self.dist_list = []
        for i in range(len(self.dist_list_tmp)):
            a = [self.dist_list_tmp[i], self.labels[i]]
            self.dist_list.append(a)      
 
        self.dist_list.sort()
        self.centroid_list = []
        
        for i in range(k):
            self.centroid_list.append(self.dist_list[i][1])

        self.total = []   
        for i in self.groups:
            a = [self.centroid_list.count(i), i]
            self.total.append(a)	
            		
        self.total.sort()
        self.total.reverse()
        self.maxvotes = self.total[0][0]
        i = 0
        try:
            while (self.total[i][0] == self.maxvotes):
                self.nn.append(self.total[i][1])
                i = i+1
        except IndexError:
        	pass
        
        return self.nn

if __name__ == '__main__':
    ts = [1,2,3,4,5]
    training_set = [[1,2,3,4,5],[5,4,3,2,1],[5,4,4,2,1],[2,2,3,4,5]]
    labels = [3,2,2,1]
    distance_type = 'dtw'
    fast = False
    radius = 20
    k = 1
    nn = kS(ts,
             training_set,
             labels,
             distance_type,
             fast,
             radius)
    print nn.compute(k)
