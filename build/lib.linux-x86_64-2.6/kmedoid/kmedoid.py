#!/usr/bin/python

__all__ = ["Medoid"]


from numpy import array, sqrt, zeros, inf,random,empty
import scipy.stats
import time
import calc_dist

class DistanceError (Exception):
	'''Distance calculation method not supported'''
	pass

class IterationError (Exception):
	'''Iteration count must be bigger than 1'''
	pass

class ClusterError (Exception):
	'''number of cluster can't be bigger than half of the total number of tseries'''
	pass


class Medoid:
	''' class that manages the algorithm of k-medoid'''
	def __init__(self, it=None, distance="ddtw", fast=False, radius=20, seed=None, tol=0.0001, pu=False):
		''' This function gets the inputs which are: nrip:number of times the cycle of clustering must be run (if not defined
		 the algorithm runs until the variants between the old and the new centroids is lower than const tol); the flag met defines
		 the method by which the distance between series is calculated (dtw/ddtw/euclidean/pearson); fast: if True use fast dtw;
		 radius: define the accurancy of fastdtw; seed: parameter for random function; tol: define the precision of the kmedoid alghoritm'''
		
		if not distance in ["dtw", "ddtw", "euclidean", "pearson"]:
			raise ValueError("distance %s is not implemented" % distance)
		if (it<1) and (it!=None):
			raise ValueError("it must be bigger than zeros" )
	
		self.nrip=it
		self.met=distance
		self.fast=fast
		self.radius=radius
		self.seed=seed		
		self.error=tol
		self.pu=pu

	def compute(self, k, mat):
		''' it gets as input: k (number of clusters) and a matrix (mat) of time series (one time series each column and a point each raw);
		 it gives the index of the medoids found as output '''

		if self.seed==None:
			random.seed(int(time.time()))
		else:
			random.seed(self.seed)
		self.k = k
		self.mat = mat.astype(float)
		self.mat=self.mat.T
		self.l=calc_dist.Dist(self.mat,self.met, self.fast, self.radius)
		if self.pu:
			self.l.pu="CPU"
		self.l
		self.r = self.mat.shape[0]
		#print "dimensione matrice",self.mat.shape
		self.medoids = zeros(k)
		self.el_clus = zeros (k)
		self.memo= zeros((self.r,self.r))
		self.memo-=1
		self.medoids-=1		
		self.min = zeros((self.r, 2))
		self.__selectmedoid()
		#print "inizializzazione"
		#print self.medoids
		if self.nrip==None:
			conf_prec=zeros(self.k)
			cond=0
			while  cond==0:
				conf_prec=self.medoids.copy()
				self.__associate()				
				#print self.min
				self.__swap()
				#print "conf_prec"
				#print conf_prec
				#print self.medoids
				cond=1
				for i in range (self.k):
					if conf_prec[i]!=self.medoids[i]:
						cond=0	
		else:
			for i in range(self.nrip):
				#print self.medoids
				self.__associate()
				#print self.min
				self.__swap()
				#print self.medoids
		matpoint=[self.mat[i] for i in self.medoids]
		self.matpoint=matpoint 
		self.min=self.min[:,0]
		self.min+=1
		self.medoids+=1
		self.medoids=self.medoids.astype(int)
		self.min=self.min.astype(int)
		return self.medoids,self.min

	def __selectmedoid(self):
		''' It gives tha array named self.medoids which contains the index of the lines of mat that contain the selected medoids'''
		for i in range(self.k):
			cond = 0			
			while cond == 0:				
				t = random.randint(self.r)
				if t in self.medoids:
					cond = 0
				else:
					self.medoids[i] = t
					cond = 1
	
	def __associate(self):
		''' It assignes each series to the nearest medoid '''
		cont=0
		li=[]	
		lista=[]
		for i in range(self.r):	
			for j in range(self.k):
				li.append((i,self.medoids[j]))
				cont+=1
				if cont>150000:
					#print ("SUPERATO LIMITEass!!!")
					cont=0
					#print "mando ",len(li)
					temp=self.__mem(li)
					lista.extend(temp)
					li=[]
				

		#print "NUMERO DTW associate",len(li)
		lista.extend(self.__mem(li))
		#print lista,li
		for i in range(self.r):
			self.min[i,1]=inf
			for j in range(self.k):
				t=lista[self.k*i+j]
				#print "%f %f %f" %(li[self.k*i+j][0],li[self.k*i+j][1],t)
				if t<self.min[i,1]:
					self.min[i,1]=t
					self.el_clus[j]+=1
					self.min[i,0]=self.medoids[j]

	def __difference_pearson (self, li):
		''' It returns the distance between 2 series computed with the Pearson correlation '''
		temp=[]
		for i in range (len(li)):
			tmp=scipy.stats.pearsonr(self.mat[li[i][0],:],self.mat[li[i][1],:])
			temp.append(tmp[1])
			#print self.mat[li[i][0],:],self.mat[li[i][1],:]
			#print tmp[1]
		return temp

	def __swap (self):
		''' for each test cluster tries to change the medoid with another series and keeps the configuration with minimum cost '''
		lista=[]
		li=[]
		cont=0
		for i in range(self.k):
			for z in range (self.r):
				if (self.min[z,0]==self.medoids[i]) and (z!=self.medoids[i]):
					li.append((self.medoids[i],z))
			for j in range (self.r):
				if (self.medoids[i]==self.min[j,0]) and (not j in self.medoids):
					for z in range (self.r):
						if (self.min[z,0]==self.medoids[i]) and (z!=j):
							li.append((j,z))
							cont+=1
							if cont>150000:
								#print ("SUPERATO LIMITEswap!!!")
								cont=0
								#print "mando ",len(li)
								temp=self.__mem(li)
								lista.extend(temp)
								li=[]
		#print "NUMERO DTW swap",len(li)				
		lista.extend(self.__mem(li))
		#print "RITORNA LUNGA ",len(lista)

		self.el_clus = zeros (self.k)
		p=0
		for i in range(self.k):
			medoid=self.medoids[i]
			old_conf=0
			for z in range (self.r):
				if (self.min[z,0]==self.medoids[i]) and (z!=self.medoids[i]):
					old_conf+=lista[p]
					p+=1
			for j in range (self.r):
				if (self.medoids[i]==self.min[j,0]) and (not j in self.medoids):
					#print "provo con la time ",j
					new_conf=0
					for z in range (self.r):
						if (self.min[z,0]==self.medoids[i]) and (z!=j):
							new_conf+=lista[p]
							p+=1
					#print "costa ",new_conf
					if new_conf<old_conf:
						#print "e' meglio, scambio"
						old_conf=new_conf
						medoid=j
			self.medoids[i]=medoid
		#print self.medoids

	def __cost (self,newmed, cluster):
		''' calculates the cost of cluster's configuration '''
#		print "chiamata a cost ",newmed,cluster
		sum_dist=0
		for j in range (self.r):
			if (self.min[j,0]==cluster) and (j!=newmed):
#				print "analizzo righe",j,newmed
				sum_dist+=self.__difference(self.mat[j,:].copy(),self.mat[newmed,:].copy())
		return sum_dist

	def __mem (self, li):
		temp=empty(len(li))
		index=[]
		for i in range(len(li)):
			if self.memo[li[i][0],li[i][1]]!=-1:
				temp[i]=self.memo[li[i][0],li[i][1]]
				li[i] = -1;				
			else:
				index.append(i)
		
		while -1 in li:
			li.remove(-1);
		if self.met!="pearson":
			ris=self.l.compute(li)
		else:
			ris=self.__difference_pearson(li)


		for i in range(len(ris)):
			self.memo[li[i][0],li[i][1]]=ris[i]
			temp[index[i]]=ris[i]

		return temp

if __name__ == "__main__":				
	k = 2	
	mat = array(   [[0,0,0,1,1,1,2,2,2,12,12,12,13,13,13,14,14,14,0,0,0,1,1,1,2,2,2],[0,1,2,0,1,2,0,1,2,7,8,9,7,8,9,7,8,9,10,11,12,10,11,12,10,11,12]] )
	m = Means(None,ddtw,False,20,None,0.0001)
	print compute(k,mat)   

