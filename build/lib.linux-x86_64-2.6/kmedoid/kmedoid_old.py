__all__ = ["Medoid"]


from numpy import array, sqrt, zeros, inf,random
import scipy.stats
import time
import dtw_cpu as dtw

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
	def __init__(self, it=None, distance="ddtw", fast=False, radius=20, seed=None, tol=0.0001):
		''' This function gets the inputs which are: nrip:number of times the cycle of clustering must be run (if not defined
		 the algorithm runs until the variants between the old and the new centroids is lower than const tol); the flag met defines
		 the method by which the distance between series is calculated (dtw/ddtw/euclidean/pearson); fast: if True use fast dtw;
		 radius: define the accurancy of fastdtw; seed: parameter for random function; tol: define the precision of the kmedoid alghoritm'''
		
		if not distance in ["dtw", "ddtw", "euclidean", "pearson"]:
			raise DistanceError("distance %s is not implemented" % distance)
		if (it<1) and (it!=None):
			raise IterationError("it must be bigger than zeros" )
	
		self.nrip=it
		self.met=distance
		self.fast=fast
		self.radius=radius
		self.seed=seed		
		self.error=tol

	def compute(self, k, mat):
		''' This function takes: number of trends, k; a matrix, mat, (where each row is a timeseries and each column a point of the time series).
Returns indices of centroids and a list which indicates the cluster each time series fit in.'''

		if self.seed==None:
			random.seed(int(time.time()))
		else:
			random.seed(self.seed)
		self.k = k
		self.mat = mat.astype(float)
		self.mat=self.mat.T
		self.r = self.mat.shape[0]
		self.medoids = zeros(k)
		self.medoids-=1		
		self.min = zeros((self.r, 2))
		if (self.k>self.r/2):
			raise ClusterError("WTF, the number of cluster can't be bigger than half of the total number of tseries" )
		
		self.__selectmedoid()
		print "inizializzazione"
		print self.medoids
		if self.nrip==None:
			conf_prec=zeros(self.k)
			cond=0
			while  cond==0:
				conf_prec=self.medoids.copy()
				self.__associate()				
				print self.min
				self.__swap()
				print self.medoids
				print "conf_prec"
				print conf_prec
				print self.medoids
				cond=1
				for i in range (self.k):
					if conf_prec[i]!=self.medoids[i]:
						cond=0	
		else:
			for i in range(self.nrip):
				print self.medoids
				self.__associate()
				self.__swap()

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
		for i in range(self.r):	
			self.min[i,1]=inf
			for j in range(self.k):
				t = self.__difference(self.mat[i].copy(), self.mat[self.medoids[j]].copy())
				print "%f %f %f" %(i,self.medoids[j],t)
				if t<self.min[i,1]:
					self.min[i,1]=t
					self.min[i,0]=self.medoids[j]
	
	def __difference (self, a, b):
		''' This fuction allows the user to choose between dtw/ddtw, euclidean distance or Pearson correlation when clustering '''
		t=0.0
		if self.met=="dtw":
			t=self.__difference_dtw (a,b,False)
		if self.met=="ddtw":
			t=self.__difference_dtw (a,b,True)				
		elif self.met=="euclidean":
			t=self.__difference_eucl (a,b)
		elif self.met=="pearson":
			t=self.__difference_pearson (a,b)
		return t

	def __difference_dtw (self, a, b, deriv):
		''' It returns the distance between 2 series calculated with the dtw algorithm '''
		temp=dtw.compute_dtw(a,b,False,deriv,self.fast,self.radius)	
		return temp

	def __difference_eucl (self, a, b):
		''' It returns the euclidean distance between 2 series '''
		val = 0 
		for i in range(self.mat.shape[1]):
			val += (a[i] - b[i])**2
		return sqrt(val)

	def __difference_pearson (self, a, b):
		''' It returns the distance between 2 series computed with the Pearson correlation '''
		t=scipy.stats.pearsonr(a, b)
		return t[1]

	def __swap (self):
		''' for each test cluster tries to change the medoid with another series and keeps the configuration with minimum cost '''
		for i in range(self.k):
			medoid=self.medoids[i]
			old_conf=self.__cost(self.medoids[i],self.medoids[i])
			#print "costa ",old_conf
			for j in range (self.r):
				if (self.medoids[i]==self.min[j,0]) and (not j in self.medoids):
					print "provo con la time ",j
					new_conf=self.__cost(j,self.medoids[i])
					print "costa ",new_conf
					if new_conf<old_conf:
						print "e' meglio, scambio"
						old_conf=new_conf
						medoid=j
			self.medoids[i]=medoid
		print self.medoids
			#print "il nuovo centroide e' ",medoid

	def __cost (self,newmed, cluster):
		''' calculates the cost of cluster's configuration '''
#		print "chiamata a cost ",newmed,cluster
		sum_dist=0
		for j in range (self.r):
			if (self.min[j,0]==cluster) and (j!=newmed):
#				print "analizzo righe",j,newmed
				sum_dist+=self.__difference(self.mat[j,:].copy(),self.mat[newmed,:].copy())
		return sum_dist

if __name__ == "__main__":				
	k = 2	
	mat = array(   [[0,0,0,1,1,1,2,2,2,12,12,12,13,13,13,14,14,14,0,0,0,1,1,1,2,2,2],[0,1,2,0,1,2,0,1,2,7,8,9,7,8,9,7,8,9,10,11,12,10,11,12,10,11,12]] )
	m = Means(None,ddtw,False,20,None,0.0001)
	print compute(k,mat)   

