#!/usr/bin/python

__all__ = ["Medoid"]


from numpy import array, sqrt, zeros, inf,random,empty
import scipy.stats
import time
import calc_dist
import dtw_cpu

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
    def __init__(self, it=None, distance="ddtw", fast=False, radius=20, seed=None, tol=0.0001, pu="CPU"):
        ''' This function gets the inputs which are: nrip:number of times the cycle of clustering must be run (if not defined
         the algorithm runs until the variants between the old and the new centroids is lower than const tol); the flag met defines
         the method by which the distance between series is calculated (dtw/ddtw/euclidean/pearson); fast: if True use fast dtw;
         radius: define the accurancy of fastdtw; seed: parameter for random function; tol: define the precision of the kmedoid alghoritm'''

        if not distance in ["dtw", "ddtw", "euclidean", "pearson"]:
            raise ValueError("distance %s is not implemented" % distance)
        if (it<1) and (it!=None):
            raise ValueError("it must be bigger than zeros" )
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
        self.met=distance
        self.fast=fast
        self.radius=radius
        self.seed=seed
        self.error=tol
        self.pu=pu

    def compute(self, k, mat):
        ''' it gets as input: k (number of clusters) and a matrix (mat) of time series (one time series each column and a point each raw);
         it gives the index of the medoids found as output '''

        self.medoids = zeros(k)
        self.medoids-=1
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
        self.l = calc_dist.Dist(self.mat,self.met, self.fast, self.radius, pu=self.pu)
        
        self.__selectmedoid()
        self.__associate()
        self.__control()
        self.__swap()
        cont_iteration=0
        if self.nrip==None:
            cond=0
            while  cond==0:
                conf_prec=self.medoids.copy()
                self.__associate()
                self.__swap()
                cond=1
                for i in range (self.k):
                    if conf_prec[i]!=self.medoids[i]:
                        cond=0
                cont_iteration+=1
                if cont_iteration>1000:
                    cond=1
        else:
            for i in range(self.nrip):
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

    def __control(self):
        cond=zeros(self.r)
        for i in range(self.r):
           cond[self.min[i,0]]+=1
        if (1 in cond):
            print "EMPTY CLUSTER:RE-INIT"
            self.reinit_maxiter+=1
            if self.reinit_maxiter>100:
                raise InitError("general error: I can't initialize the clusters, try changing distance measure or provide other dataset" )
            self.__selectmedoid()
            self.__associate()
            self.__control()

    def __associate(self):
        ''' It assignes each series to the nearest medoid '''
        if self.pu=="GPU":
            self.__associate_gpu()
        else:
            self.__associate_cpu()

    def __associate_cpu(self):
        ''' It assignes each series to the nearest medoid '''
        for i in range(self.r):    
            self.min[i,1]=inf
            for j in range(self.k):
                t = self.__difference(self.mat[i].copy(), self.mat[self.medoids[j]].copy())
                if t<self.min[i,1]:
                    self.min[i,1]=t
                    self.min[i,0]=self.medoids[j]

    def __difference (self, a, b):
        ''' This fuction allows the user to choose between dtw/ddtw, euclidean distance or Pearson correlation when clustering '''
        t=0.0
        if (self.met=="dtw") or (self.met=="ddtw"):
            t=self.__difference_dtw (a,b)
        elif self.met=="euclidean":
            t=self.__difference_eucl (a,b)
        elif self.met=="pearson":
            t=self.__difference_pearson (a,b)
        return t

    def __difference_dtw(self, a, b):
        ''' It returns the distance between 2 series calculated with the dtw algorithm '''
        if self.met=="ddtw":
            derivative=True
        else:
            derivative=False
        temp=dtw_cpu.compute_dtw(a, b, False, False, derivative, self.fast, self.radius)
        return temp

    def __difference_eucl (self, a, b):
        ''' It returns the euclidean distance between 2 series '''
        val = 0 
        for i in range(self.mat.shape[1]):
            val += (a[i] - b[i])**2
        return val

    def __difference_pearson (self, a, b):
        ''' It returns the distance between 2 series computed with the Pearson correlation '''
        t=scipy.stats.pearsonr(a, b)
        return t[1]

    def __associate_gpu(self):
        cont=0
        li=[]
        lista=[]
        for i in range(self.r):
            for j in range(self.k):
                li.append((i,self.medoids[j]))
                cont+=1
                if cont>150000:
                    cont=0
                    temp=self.__mem(li)
                    lista.extend(temp)
                    li=[]

        lista.extend(self.__mem(li))
 
        for i in range(self.r):
            self.min[i,1]=inf
            for j in range(self.k):
                t=lista[self.k*i+j]
                if t<self.min[i,1]:
                    self.min[i,1]=t
                    self.min[i,0]=self.medoids[j]

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

    def __swap (self):
        if self.pu=="GPU":
            self.__swap_gpu()
        else:
            self.__swap_cpu()

    def __swap_gpu (self):
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
                                cont=0
                                temp=self.__mem(li)
                                lista.extend(temp)
                                li=[]

        lista.extend(self.__mem(li))

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
                    new_conf=0
                    for z in range (self.r):
                        if (self.min[z,0]==self.medoids[i]) and (z!=j):
                            new_conf+=lista[p]
                            p+=1
                    if new_conf<old_conf:
                        old_conf=new_conf
                        medoid=j
            self.medoids[i]=medoid

    def __swap_cpu (self):
        for i in range(self.k):
            medoid=self.medoids[i]
            old_conf=0
            for z in range (self.r):
                if (self.min[z,0]==self.medoids[i]) and (z!=self.medoids[i]):
                    old_conf+=self.__difference(self.mat[self.medoids[i]].copy(),self.mat[z].copy())
            for j in range (self.r):
                if (self.medoids[i]==self.min[j,0]) and (not j in self.medoids):
                    new_conf=0
                    for z in range (self.r):
                        if (self.min[z,0]==self.medoids[i]) and (z!=j):
                            new_conf+=self.__difference(self.mat[j].copy(),self.mat[z].copy())
                    if new_conf<old_conf:
                        old_conf=new_conf
                        medoid=j
            self.medoids[i]=medoid
                
        
if __name__ == "__main__":
    k = 2
    mat = array(   [[0,0,0,1,1,1,2,2,2,12,12,12,13,13,13,14,14,14,0,0,0,1,1,1,2,2,2],[0,1,2,0,1,2,0,1,2,7,8,9,7,8,9,7,8,9,10,11,12,10,11,12,10,11,12]] )
    m = Means(None,ddtw,False,20,None,0.0001)
    print compute(k,mat)

