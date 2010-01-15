#!/usr/bin/python

import calc_dist as cd
import numpy as np

class kNN:

    def __init__(self, test_set, training_set, label_list, weight=True,
                 distance_type="dtw", fast=False, radius=20, pu="CPU"):
        self.test_set = test_set
        self.training_set = training_set
        self.label_list = label_list
        self.weight = weight
        self.distance_type = distance_type
        self.fast = fast
        self.radius = radius
        self.pu=pu
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

    def compute(self, k=1):
        self.k = k
        self.nn = []
        self.dist_list = []
        self.label_dict = {}
        self.keys = []
        self.ts_label = 0

        self.dist_list_tmp = cd.Dist(self.tmp_matrix,
                                     self.distance_type,
                                     self.fast,
                                     self.radius,
                                     self.pu).compute(self.calc_list)

        for i in range(len(self.dist_list_tmp)):
        #from numpy array to list
            self.dist_list.append(self.dist_list_tmp[i])

        if self.weight == True:
            for i in range(self.k):
                self.nn.append((self.label_list[self.dist_list.index(min(self.dist_list))], \
                min(self.dist_list)))
                self.dist_list[self.dist_list.index(min(self.dist_list))] = np.inf

            for i in range(self.k):
                if self.label_dict.__contains__(self.nn[i][0]) == False:
                    self.label_dict[self.nn[i][0]] = [1,self.nn[i][1],1/self.nn[i][1]]
                    self.keys.append(self.nn[i][0])
                else:
                    self.label_dict[self.nn[i][0]][0] = self.label_dict[self.nn[i][0]][0]+1
                    self.label_dict[self.nn[i][0]][1] = self.label_dict[self.nn[i][0]][1]+self.nn[i][1]
                    self.label_dict[self.nn[i][0]][2] = self.label_dict[self.nn[i][0]][0] / \
                    self.label_dict[self.nn[i][0]][1]
            for i in range(len(self.keys)):
                if self.ts_label == 0 or self.label_dict[self.keys[i]][2] > self.label_dict[self.ts_label][2]:
                    self.ts_label = self.keys[i]
                else:
                    pass

        else:
            print 'you better use the weighted method you motherfucker!!!'

        return self.ts_label


if __name__ == '__main__':
    test_set = [1,2,3,4,5]
    training_set = [[1,2,3,4,5],[5,4,3,2,1],[5,4,4,2,1],[2,2,3,4,5]]
    label_list = ['a','b','b','a']
    distance_type = 'dtw'
    fast = False
    radius = 20
    k = 4
    weight = True
    nn = kNN(test_set,
             training_set,
             label_list,
             weight,
             distance_type,
             fast,
             radius)
    print nn.compute(k)
