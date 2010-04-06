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
        self.pu = pu

        if len(self.training_set) != len(self.label_list):
            raise ValueError

    def compute(self, k=1):
        self.k = k
        self.test_set_labels = []

    	if self.k > len(self.training_set) or self.k < 1:
    		raise ValueError

        for i in range(len(self.test_set)):
            tmp_set = []
            calc_list = []
            self.dist_list = []
            tmp_set.extend(self.training_set)
            tmp_set.append(self.test_set[i])
            for j in range(len(tmp_set) - 1):
                calc_list.append((len(tmp_set) - 1, j))
            self.dist_list_tmp = cd.Dist(tmp_set,
                                         self.distance_type,
                                         self.fast,
                                         self.radius,
                                         self.pu).compute(calc_list)
            for j in range(len(self.dist_list_tmp)):
            #from numpy array to list
                self.dist_list.append((self.dist_list_tmp[j],j))
            self.test_set_labels.append(self.run())

        return self.test_set_labels

    def run(self):
        nn = []
        label_dict = {}
        keys = []
        ts_label = 0
        self.dist_list.sort()

        if self.weight == True:
            for i in range(self.k):
                nn.append((self.label_list[self.dist_list[i][1]], self.dist_list[i][0]))

            for i in range(self.k):
                if label_dict.__contains__(nn[i][0]) == False:
                    label_dict[nn[i][0]] = [1,nn[i][1],1/nn[i][1]]
                    keys.append(nn[i][0])
                else:
                    label_dict[nn[i][0]][0] = label_dict[nn[i][0]][0] + 1
                    label_dict[nn[i][0]][1] = label_dict[nn[i][0]][1] + nn[i][1]
                    label_dict[nn[i][0]][2] = label_dict[nn[i][0]][0] / \
                    label_dict[nn[i][0]][1]
            for i in range(len(keys)):
                if ts_label == 0 or label_dict[keys[i]][2] > label_dict[ts_label][2]:
                    ts_label = keys[i]
                else:
                    pass

        else:
            for i in range(self.k):
                nn.append(self.label_list[self.dist_list[i][1]])
            for i in range(self.k):
                if label_dict.__contains__(nn[i][0]) == False:
                    label_dict[nn[i][0]] = 1
                    keys.append(nn[i][0])
                else:
                    label_dict[nn[i][0]] = label_dict[nn[i][0]]+1
            for i in range(len(keys)):
                if ts_label == 0 or label_dict[keys[i]] > label_dict[ts_label]:
                    ts_label = keys[i]


        return ts_label


if __name__ == '__main__':
    test_set = [[1,2,3,4,5],[5,4,3,2,1],[2,2,3,4,5],[5,4,4,2,1]]
    training_set = [[1,2,3,4,5],[5,4,3,2,1],[5,4,4,2,1],[2,2,3,4,5]]
    label_list = ['1','2','2','1']
    distance_type = 'dtw'
    fast = False
    radius = 20
    k = 4
    weight = False
    nn = kNN(test_set,
             training_set,
             label_list,
             weight,
             distance_type,
             fast,
             radius)
    print nn.compute(k)
