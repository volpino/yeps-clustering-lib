#!/usr/bin/python

import calc_dist as cd
import numpy as np


class kNN:



	def __init__(self, ts, training_set, labels, centroids, distance_type="dtw", fast=False, radius=20):
		self.ts=ts
		self.training_set=training_set
		self.labels=labels
		self.centroids=centroids
		self.distance_type=distance_type
		self.fast=fast
		self.radius=radius
		self.calc_list=[]

		self.tmp_matrix=training_set.append(ts)

		cd.Dist.__init__(self.tmp_matrix, self.ditsance_type, self.fast, self.radius) 

		for i in range(len(self.tmp_matrix)-1):
			self.calc_list.append((len(self.tmp_matrix)-1,i))

		self.dist_list=cd.Dist.compute(self.calc_list) #is this right?!


	def compute(self, k=1):

		self.k=k
		self.label_list=[]
		self.label_dict={}
		self.nn=[]

		if k==1:
			if self.dist_list.count(min(self.dist_list))==1:
				self.nn.append(self.labels[self.dist_list.index(min(self.dist_list))])
			else:
				for i in range(self.dist_list.count(min(self.dist_list))):
					self.nn.append(self.labels[self.dist_list.index(min(self.dist_list))])
					self.dist_list[self.dist_list.index(min(self.dist_list))]=np.inf
				for j in range(len(self.nn-1)):
					if self.nn[j]==self.nn[j+1]:
						self.nn.__delitem__(j+1)
					else:
						continue

		else:
			for i in range(self.k):
				self.label_list.append(self.dist_list.index(min(self.dist_list)))
				self.dist_list[self.dist_list.index(min(self.dist_list))]=np.inf
			for j in range(len(self.centroids)):
				self.label_dict[str(self.centroids[j])] = self.dist_list.count(self.centroid[j])
			while self.nn==[] or self.nn>max(self.label_dict, key=lambda x:self.label_dict.get(x)):
				self.nn.append(max(self.label_dict, key=lambda x:self.label_dict.get(x)))
			


		return nn


if __name__ == '__main__':
	
	ts = [1,2,3,4,5]
	training_set = [[1,2,3,4,5],[5,4,3,2,1],[5,4,4,2,1],[2,2,3,4,5]]
	centroids = [1,2]
	labels = [1,2,2,1]
	k = 1
	distance_type = 'dtw'
	fast = False
	radius = 20

	dist_list=__init__(ts, training_set, labels, centroids, distance_type, fast, radius)
	nn = compute(k)
	print nn
