#!/usr/bin/python

import calc_dist as cd
import numpy as np

class kS:

    def __init__(self, ts, training_set, labels, centroids,
                 distance_type="dtw", fast=False, radius=20, pu="CPU"):
        self.ts = ts
        self.training_set = training_set
        self.labels = labels
        self.centroids = centroids
        self.distance_type = distance_type
        self.fast = fast
        self.radius = radius
        self.calc_list = []
        self.pu = pu

        self.tmp_matrix = []
        for i in range(len(self.training_set)):
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
            a = [self.labels[i],self.dist_list_tmp[i]]
            self.dist_list.append(a)      
        
        self.dist_list.sort()
        self.centroids.sort()
       	self.dist_list = zip(*self.dist_list)           
        self.cluster_averages = []
        for i in range(len(self.centroids)):
            start = self.dist_list[0].index(self.centroids[i])
            try:
                end = self.dist_list[0].index(self.centroids[i+1])
            except IndexError:
                end = len(self.dist_list[0])
            avr = sum(self.dist_list[1][start:end]) / len(self.dist_list[1][start:end])
            self.cluster_averages.append([avr, self.centroids[i]])
        
        self.cluster_averages.sort()
        self.mindist = self.cluster_averages[0][0]
        i = 0
        try:
            while (self.cluster_averages[i][0] == self.mindist):
                self.nn.append(self.cluster_averages[i][1])
                i = i+1
        except IndexError:
        	pass
        return self.nn

if __name__ == '__main__':
    ts = [1,2,3,4,5]
    training_set = [[1,2,3,4,5],[5,4,3,2,1],[5,4,4,2,1],[2,2,3,4,5]]
    centroids = [1,2,3]
    labels = [3,2,2,1]
    distance_type = 'dtw'
    fast = False
    radius = 20
    nn = kS(ts,
             training_set,
             labels,
             centroids,
             distance_type,
             fast,
             radius)
    print nn.compute()
