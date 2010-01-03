#!/usr/bin/env python

from numpy import inf

class DTW:
	'''
Implementation of the DTW algorithm, using a recursive function to calculate the minimum path.
The recursive function starts from the left bottom edge of the matrix and calculates recursively the min path for adiacent cells. In each cell it stores the minimum path returned by adiacent cells. The calculation stops when the top right edge is reached by all of the recoursion launched.
	
Takes two time series as input and returns the minimum path and it's lenght as output (the path is returned only if the pathflag is set to 1)
	'''
	
	def __init__(self, s1, s2):
		''' Initilalizing time series '''
		self.s1 = s1
		self.s2 = s2

	def deriv_calc(self, series):
		t = series[0]
		for i in range(1,len(series)-1):
			t1 = series[i]
			series[i] = ((series[i] - t) + ((series[i+1] -t )/2))/2
			t = t1
		series[0] = series[1] - (series[2] - series[1])
		series[-1] = series[-2] - (series[-3] - series[-2])
		return series
			
	def dist_calc(self, deriv):
		''' Calculates the distance between every point of the two time series and stores it into the distance matrix '''
		if deriv == 1:
			self.s1 = self.deriv_calc(self.s1)
			self.s2 = self.deriv_calc(self.s2)

		self.dist = [] #allocation of the matrix
		for i in range(self.s1.__len__()):
			self.dist.append([j for j in range(self.s2.__len__())])
		for i in enumerate(self.s1): #filling the matrix with distances
			for j in enumerate(self.s2):
				self.dist[i[0]][j[0]] = ((i[0] - j[0])**2 + (i[1] - j[1])**2)
	def short_path(self, pathflag):
		''' Calculates the minimum path in the distance matrix, recursively with memorizization. '''
		self.mem = [] # list for storing already-calculated paths
		for i in range(self.s1.__len__()):
			self.mem.append([-1 for j in range(self.s2.__len__())]) #initializing it to -1 (-1 means not calculated yet) 
		def next(x,y): #recursive function
			if x == self.s1.__len__() -1 and y == self.s2.__len__()-1: #if we've reached the end, return.
				return self.dist[x][y]
			try:
				if self.mem[x][y] != -1: #if the value has been already calculated for this cell return it without recomputing it
					return self.mem[x][y]
			except IndexError: # if x or y exceed the matrix bounds return inf
				return inf
			a = next(x+1,y);	#calculate min path for near cells
			b = next(x+1, y+1);
			c = next(x, y+1);
			self.mem[x][y] = min(a,b,c) + self.dist[x][y] #sum the value of this cell and the minimum value of near cells, store and return it
			return self.mem[x][y]

		a = next(0,0)
		if pathflag != 1: return a

		pathlist = []
		x = 0
		y = 0
		while 1:
			pathlist.append([x,y])
			if (x == self.s1.__len__()-1) and (y == self.s2.__len__()-1): break
			if x == self.s1.__len__()-1: 
				y += 1
				continue
			if y == self.s2.__len__()-1:
				x +=1
				continue
			if (self.mem[x+1][y+1] < self.mem[x][y+1]) and (self.mem[x+1][y+1] < self.mem[x+1][y]):
				x += 1
				y += 1
				continue
			if (self.mem[x+1][y] < self.mem[x][y+1]) and (self.mem[x+1][y] < self.mem[x+1][y+1]):
				x += 1
				continue
			y += 1;

		return [a, pathlist];

	def run(self, deriv = 0, pathflag = 0):
		''' Runs the algorithm and returns results
	If pathflag is 1 it returns a list containing the min distance an a list of point representing the path, else it returns only the distance '''
		self.dist_calc(deriv)
		return self.short_path(pathflag)



if __name__ == "__main__":
	a = DTW([1.,2,1,-4,7,3,2,9],[8,5.,7,9,9,1,2,3])
	b = a.run(1,1)
	print b[0]
	print b[1]
	


