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
        
        if len(training_set) != len(labels):
            raise ValueError
        
        self.groups = []
        self.groups.extend(self.labels)
        self.groups.sort()
        last = self.groups[-1]
        for i in range(len(self.groups)-2, -1, -1):
            if last==self.groups[i]: 
                del self.groups[i]
            else: last=self.groups[i]
            
        
    def compute_single(self, test, k):
    	if k > len(self.training_set):
    		raise ValueError

        tmp_matrix = []
        for i in range(len(self.training_set)):
            tmp_matrix.append(self.training_set[i])
        tmp_matrix.append(test)
        
        calc_list = []
        for i in range(len(tmp_matrix) - 1):
            calc_list.append([len(tmp_matrix) - 1, i])

        dist_list_tmp = cd.Dist(tmp_matrix,
                             self.distance_type,
                             self.fast,
                             self.radius,
                             self.pu).compute(calc_list)
        nn=[]  
        dist_list = []
        for i in range(len(dist_list_tmp)):
            a = [dist_list_tmp[i], self.labels[i]]
            dist_list.append(a)      
 
        dist_list.sort()
        centroid_list = []
        
        for i in range(k):
            centroid_list.append(dist_list[i][1])

        total = []   
        for i in self.groups:
            a = [centroid_list.count(i), i]
            total.append(a)

        total.sort()
        total.reverse()
        maxvotes = total[0][0]
        i = 0
        try:
            while (total[i][0] == maxvotes):
                nn.append(total[i][1])
                i = i+1
        except IndexError:
        	pass
        return nn

    def compute(self, k=1):
        ret = []
        for i in self.ts:
            ret.append(self.compute_single(i, k))

        return ret

if __name__ == '__main__':
    ts = [[1,2,3,4,5],[5,4,3,2,1],[4,5,6,2,5]]
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
