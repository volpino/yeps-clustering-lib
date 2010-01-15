#!/usr/bin/python

import calc_dist as cd
import numpy as np

class kNN:

    def __init__(self, test_set, training_set, label_list,
                 distance_type="dtw", fast=False, radius=20, pu="CPU"):
        self.test_set = test_set
        self.training_set = training_set
        self.label_list = label_list
        self.distance_type = distance_type
        self.fast = fast
        self.radius = radius
        self.labels = []
        self.calc_list = []
        self.tmp_matrix = []

        for i in range(len(self.label_list)):
            if self.labels.__contains__(self.label_list[i]) == False:
                self.labels.append(self.label_list[i])
            else:
                pass

        for i in range(len(self.training_set)):
            self.tmp_matrix.append(self.training_set[i])
        self.tmp_matrix.append(self.test_set)

        for i in range(len(self.tmp_matrix) - 1):
            self.calc_list.append((len(self.tmp_matrix) - 1, i))

        self.dist_list_tmp = cd.Dist(self.tmp_matrix,
                                     self.distance_type,
                                     self.fast,
                                     self.radius,
                                     pu=pu).compute(self.calc_list)

    def compute(self, k=1):
        self.k = k
        self.nn = []
        self.dist_list = []
        self.label_dict = {}
        self.ts_label = []

        for i in range(len(self.dist_list_tmp)):
        #from numpy array to list
            self.dist_list.append(self.dist_list_tmp[i])

        for i in range(self.k):
            for j in range(self.dist_list.count(min(self.dist_list))):
                self.nn.append(self.label_list[self.dist_list.index(min(self.dist_list))])
                self.dist_list[self.dist_list.index(min(self.dist_list))]=np.inf

        for i in range(len(self.label_list)):
            self.label_dict[self.label_list[i]] = self.dist_list.count(self.label_list[i])


        self.ts_label.append(max(self.label_dict, key=lambda x:self.label_dict.get(x)))


        return self.ts_label


if __name__ == '__main__':
    test_set = [1,2,3,4,5]
    training_set = [[1,2,3,4,5],[5,4,3,2,1],[5,4,4,2,1],[2,2,3,4,5]]
    labels = ['a','b','b','a']
    distance_type = 'dtw'
    fast = False
    radius = 20
    k = 1
    nn = kNN(test_set,
             training_set,
             labels,
             distance_type,
             fast,
             radius)
    print nn.compute(k)
