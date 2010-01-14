#!/usr/bin/python

import calc_dist as cd
import numpy as np

class kS:

    def __init__(self, ts, training_set, centroids,
                 distance_type="dtw", fast=False, radius=20, pu="CPU"):
        self.ts = ts
        self.training_set = training_set
        self.centroids = centroids
        self.distance_type = distance_type
        self.fast = fast
        self.radius = radius
        self.calc_list = []
        self.pu = pu
        
    def compute_single(self, test):
    
        tmp_matrix = []
        for i in self.centroids:
            tmp_matrix.append(self.training_set[i])
        tmp_matrix.append(test)

        calc_list = []
        for i in range(len(tmp_matrix) - 1):
            calc_list.append((len(tmp_matrix) - 1, i))
    
        dist_list_tmp = cd.Dist(tmp_matrix,
                             self.distance_type,
                             self.fast,
                             self.radius,
                             self.pu).compute(calc_list)
        nn=[]  
        dist_list = []
        for i in range(len(dist_list_tmp)):
            a = [dist_list_tmp[i], self.centroids[i]]
            dist_list.append(a)      
        
        dist_list.sort()
        mindist = dist_list[0][0]
        i = 0
        try:
            while (dist_list[i][0] == mindist):
                nn.append(dist_list[i][1])
                i = i+1
        except IndexError:
        	pass
        return nn
        
    def compute(self):    
        ret = []
        for i in ts:
            ret.append(self.compute_single(i))

        return ret

if __name__ == '__main__':
    ts = [[1,2,3,4,5],[5,4,3,2,1],[4,5,6,2,5]]
    training_set = [[1,2,3,4,5],[5,4,3,2,1],[5,4,4,2,1],[2,2,3,4,5]]
    centroids = [0,3,1]
    distance_type = 'dtw'
    fast = False
    radius = 20
    nn = kS(ts,
             training_set,
             centroids,
             distance_type,
             fast,
             radius)
    print nn.compute()
