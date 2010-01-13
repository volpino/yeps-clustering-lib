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

        self.tmp_matrix = []
        for i in self.centroids:
            self.tmp_matrix.append(self.training_set[i])
        self.tmp_matrix.append(self.ts)

        for i in range(len(self.tmp_matrix) - 1):
            self.calc_list.append((len(self.tmp_matrix) - 1, i))
        
    def compute(self):
        self.dist_list_tmp = cd.Dist(self.tmp_matrix,
                             self.distance_type,
                             self.fast,
                             self.radius,
                             self.pu).compute(self.calc_list)
        self.nn=[]  
        self.dist_list = []
        for i in range(len(self.dist_list_tmp)):
            a = [self.dist_list_tmp[i], self.centroids[i]]
            self.dist_list.append(a)      
        
        self.dist_list.sort()
        self.mindist = self.dist_list[0][0]
        i = 0
        try:
            while (self.dist_list[i][0] == self.mindist):
                self.nn.append(self.dist_list[i][1])
                i = i+1
        except IndexError:
        	pass
        return self.nn

if __name__ == '__main__':
    ts = [1,2,3,4,5]
    training_set = [[1,2,3,4,5],[5,4,3,2,1],[5,4,4,2,1],[2,2,3,4,5]]
    centroids = [1,2,3]
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
