#!/usr/bin/python


        

from numpy import array, zeros, ones, inf,random,empty
import scipy.stats
import dtw_cpu
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

class InitError (Exception):
    '''general error: I can't initialize the clusters, try changing distance measure or provide other dataset'''
    pass

class Means:
    ''' class that manages the algorithm of k-means'''

    def __init__(self, it=None, distance="ddtw", fast=False, radius=20, seed=None, tol=0.0001, pu="CPU"):
        '''
        This function gets the inputs which are: nrip:number of times the
        cycle of clustering must be run (if not defined the algorithm runs
        until the variants between the old and the new centroids is lower
        than const tol); the flag met defines the method by which the
        distance between series is calculated (dtw/ddtw/euclidean/pearson);
        fast: if True use fast dtw;
        radius: define the accurancy of fastdtw; seed: parameter for
        random function; tol: define the precision of the kmedoid alghoritm
        '''
        if not distance in ["dtw", "ddtw", "euclidean", "pearson"]:
            raise DistanceError("distance %s is not implemented" % distance)
        if (it<1) and (it!=None):
            raise IterationError("it must be bigger than zeros" )
        if pu == "GPU":
            try:
                from dtw_gpu import dtw_gpu
            except ImportError:
                print "No suitable hardware! Doing DTW on CPU..."
                pu = "CPU"
        if (pu=="GPU"):
            if (distance!="ddtw") and (distance!="dtw"):
                print "Il calcolo su Gpu e' sisponibile solo con dtw/ddtw"
                distance="dtw"

        self.nrip=it
        self.distance=distance
        self.fast=fast
        self.radius=radius    
        self.seed=seed
        self.error=tol
        self.pu=pu
        
       
    def compute (self, k, mat):
        '''
        This function takes: number of trends, k; a matrix, mat,
        (where each row is a timeseries and each column a point of the time
        series).
        Returns indices of centroids and a list which indicates the cluster
        each time series fit in.
        '''
        if (k>mat.shape[1]/2):
            raise ClusterError("The number of cluster can't be bigger than half of the total number of tseries" )
        
        self.centroids = zeros(k)
        self.centroids-=1
        self.memo= zeros((mat.shape[1],mat.shape[1]))
        self.memo-=1
        self.min = zeros((mat.shape[1], 2))
        self.reinit_maxiter=0
        if self.seed==None:
            random.seed(int(time.time()))
        else:
            random.seed(self.seed)

        self.k = k
        self.mat = mat.astype(float)
        self.mat=self.mat.T
        self.r = self.mat.shape[0]
        self.l=calc_dist.Dist(self.mat,self.distance, self.fast, self.radius, self.pu)
        
        self.__select_centroids() # calls the function that assigns random centroids
        self.__compare()    # calls the function that puts each time series with the most similar centroid
        self.__control()    # calls the function that checks that no empty clusters came out from the random choice
        old_error = self.__calc_err() # calls the function  thet calculates the error
        new_error = old_error*(2.0+self.error)
        cont=0
        if not self.nrip:
            while (abs(new_error/old_error-1.0)>self.error) and (cont<500): 
                self.__newcentroids()   # calls the function that calculates new centroids
                self.__compare()
                cont+=1
                old_error=new_error
                new_error=self.__calc_err()
        else:
            for i in range(self.nrip-1):
                self.__newcentroids()
                self.__compare()
       
        self.min=self.min[:,0]
        self.min+=1
        self.centroids+=1
        self.centroids=self.centroids.astype(int)
        self.min=self.min.astype(int)
        return self.centroids,self.min


    def __select_centroids(self):
        ''' It gives tha array named self.centroids which contains the index of the lines of mat that contain the selected centroids'''
        for i in range(self.k):
            cond = 0
            while cond == 0:                # cycle that picks k different numbers randomly
                t = random.randint(self.r)
                if t in self.centroids:
                    cond = 0
                else:
                    self.centroids[i] = t
                    cond = 1


    def __compare(self):
        if self.pu=="GPU":
            self.__compare_gpu()
        else:
            self.__compare_cpu()
 
       
    def __compare_cpu(self):
        ''' It assignes each series to the nearest centroid '''
        for i in range(self.r):	# cycle that scrolls every time series
            listdiff = zeros(self.k)
            for j in range(self.k):
                listdiff[j] = self.__difference(self.mat[i].copy(), self.mat[self.centroids[j]].copy()) # records in listdiff the distance between the time series i and the centroid self.centroids[j]
            val=inf
            for j in range(self.k):
                if val > listdiff[j]:
                    val=listdiff[j]
                    self.min[i,0] = self.centroids[j]	# self.min is a matrix that contains a line for each time series, in the first it's recorded the centroid from which it is closer and in the second the distance from that centroid
                    self.min[i,1] = val


    def __difference(self, a, b):
        ''' This fuction allows the user to choose between dtw/ddtw, euclidean distance or Pearson correlation when clustering '''
        t=0.0
        if (self.distance=="dtw") or (self.distance=="ddtw"):
            t=self.__difference_dtw (a,b)
        elif self.distance=="euclidean":
            t=self.__difference_eucl (a,b)
        elif self.distance=="pearson":
            t=self.__difference_pearson (a,b)
        return t


    def __difference_dtw(self, a, b):
        ''' It returns the distance between 2 series calculated with the dtw algorithm '''
        if self.distance=="ddtw":
            derivative=True
        else:
            derivative=False
        temp=dtw_cpu.compute_dtw(a, b, False, False, derivative, self.fast, self.radius)
        return temp


    def __difference_eucl(self, a, b):
        ''' It returns the euclidean distance between 2 series '''
        val = 0
        for i in range(self.mat.shape[1]):
            val += (a[i] - b[i])**2
        return val

    
    def __difference_pearson (self, a, b):
        ''' It returns the distance between 2 series computed with the Pearson correlation '''
        t=scipy.stats.pearsonr(a, b)
        return t[1]


    def __compare_gpu(self):
        ''' It assignes each series to the nearest centroid '''
        li=[]
        lista=[]
        cont=0
        for i in range(self.r): # cycle that scrolls every time series
            for j in range(self.k):
                li.append((i, self.centroids[j]))
                cont+=1
                if cont>150000:
                    cont=0
                    temp=self.__mem(li)
                    lista.extend(temp)
                    li=[]

        lista.extend(self.__mem(li))

        for i in range(self.r): # cycle that scrolls every time series
            minimum=inf
            for j in range(self.k):
                if (minimum>=lista[i*self.k+j]):
                    self.min[i,0] = self.centroids[j]
                    self.min[i,1] = lista[i*self.k+j]
                    minimum=lista[i*self.k+j]


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

        ris=self.l.compute(li)

        for i in range(len(ris)):
            self.memo[li[i][0],li[i][1]]=ris[i]
            temp[index[i]]=ris[i]

        return temp


    def __control(self):
        cond=zeros(self.r)
        for i in range(self.r):
           cond[self.min[i,0]]+=1
        if (1 in cond):
            print "EMPTY CLUSTER:RE-INIT"
            self.reinit_maxiter+=1
            if self.reinit_maxiter>100:
                raise InitError("general error: I can't initialize the clusters, try changing distance measure or provide other dataset" )
            self.__select_centroids()
            self.__compare()
            self.__control()


    def __calc_err(self):
        ''' It calculates the error which is the difference between the variances of the previous and the current cycle '''
        varianza = zeros (self.k)
        for i in range(self.k): # cycle that calculates the variance of each centroid
            div=0
            for j in range(self.r):
                if self.min[j,0]==self.centroids[i]:
                    varianza[i]+=self.min[j,1]**2
                    div+=1
            varianza[i]=varianza[i]/div
        return  varianza.sum(axis=0)


    def __newcentroids(self):
        ''' It finds a new set of centroids: for each cluster it picks as new centroid the time series which is closest to the average of the distances from the old centroid and the other series in that cluster '''
        for i in range(self.k):
            media = zeros(self.mat.shape[1])
            index = 0
            for j in range(self.r): # average calculation
                if self.min[j,0] == self.centroids[i]:
                    index+=1
                    for k in range(self.mat.shape[1]):
                        media[k] += self.mat[j,k]

            media /= index
            mini = inf
            for j in range(self.r):     # finds the time series wich has the closest distance to the average of the distances from the centroid to each time series in that cluster
                if self.min[j,0] == self.centroids[i]:
                    dif=self.__difference(self.mat[j].copy(),media.copy())
                    if mini>dif:
                        mini = dif
                        centroi = j
            self.centroids[i]=centroi



if __name__ == "__main__":
    k = 2   # number of clusters the time series have to be divided in
    mat = array(   [[4,2.1,6,1],[4.2,3,6,6],[1,2.1,4,4],    # each line is a time series
                     [7,5,7,4],[8,6,5.7,2],[7,8.1,9,1],
                    [1,2,4,4],[7,5,9,7.4],[3,1.2,4,5]] )
    m = Means()
    a,b=m.compute(k.mat)

