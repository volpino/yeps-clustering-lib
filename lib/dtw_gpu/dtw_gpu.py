#!/usr/bin/env python
class _DTW_:
	try:
		import pycuda.driver as drv
		import pycuda.compiler as compiler
		import pycuda.autoinit
		#drv.init()
                #dev=drv.Device(1)
                #ctx=dev.make_context()
	except:
		raise ImportError("Cannot find PyCUDA")
	import numpy
	#from sys import exit
	#exit()
	def compute_dtw(self, li):
		#print "START - ", len(li), " dtw of ", len(self.matrix[0]), " elements -"
		def ololol(self, li):
			res = None
			c = 0
			while res==None:
				if c!=0:
					print "[OMG] Oh my RAM :O (trying with needed_threads = ", self.needed_threads, ")"
					self.needed_threads-=int(75*self.needed_threads/self.s)
				li_=li[:self.needed_threads*self.blocks]
				#print "len(li): ", len(li_), "\nblocks=", self.blocks, "\nthread_per_blocks=", self.needed_threads, "\n\n"
				res = self.olo_(li_)
				c+=1
			return res
		self.f = self.f_c.get_function("compute_dtw")
		self.f_euclidean = self.f_c.get_function("EUCLIDEAN")
		atts = [(str(att), value) for att, value in self.drv.Device(0).get_attributes().iteritems()]
                atts.sort()
                atts_ = {}
                for attribute, value in atts: atts_[attribute] = value
                #print atts_
                a = self.drv.Device(0).total_memory()
                n_float = a / 4 # (a*8)/32
                n_blocks = atts_['MULTIPROCESSOR_COUNT']
                n_threads = atts_['MAX_THREADS_PER_BLOCK']
                #print "blocks = ", n_blocks, "\nthreads = ", n_threads
		res = self.numpy.array([])
		while len(li)>0:
               		for i in range(n_blocks, 0, -1):
				#print len(li), "/", i , " = ", len(li)/i, " (resto di ", len(li)%i,")"
                        	if len(li)%i == 0: break
			self.blocks = i
			self.needed_threads = len(li)/self.blocks
			#print "[D] N = ", self.blocks, ", T = ", self.needed_threads
			if self.needed_threads < n_threads:
				#print "<>", n_threads
				tmp = ololol(self, li)
			 	res = self.numpy.append(res, tmp)
				break
			else:
				self.blocks_ = self.blocks
				self.needed_threads_ = self.needed_threads
				while self.needed_threads > n_threads:
					for i in range(n_blocks, 0, -1):
                                		if len(li)%i == 0: break
        	        	        self.blocks = i
	        	                self.needed_threads = len(li)/self.blocks
					#print self.blocks, self.needed_threads
					if self.blocks_ == self.blocks and self.needed_threads_ == self.needed_threads:
						self.needed_threads = n_threads
						break
				tmp = ololol(self, li)
				res = self.numpy.append(res, tmp)
				li = li[n_threads*self.blocks:]
		#self.ctx.pop()
		return res
	def olo_(self, li):
		#print li
		#print "INPUT: ", self.needed_threads , "x", self.blocks, "(", len(li), ")"
		a = self.drv.Device(0).total_memory()
		n_float = a / 4 # (a*8)/32
		res=self.numpy.empty([1, len(li)])
		len_li=self.numpy.array([len(self.matrix[0])])
		tmp=[]
		for qui in li: # nocomment -.-'
			s1 = self.matrix[qui[0]]
			s2 = self.matrix[qui[1]]
			for i in s1:
				tmp.append(i)
			for i in s2:
				tmp.append(i)
		series=self.numpy.array(tmp)
		matrix = len(self.matrix[li[0][0]])*4*len(li)
		#print "=> ", matrix
		matrix=self.numpy.empty([1, matrix])
		if self.derivative:
			deriv = 1.0
		else:
			deriv = 0.0
		deriv=self.numpy.array([deriv])
		len_in=len_li
		len_in=len_in.astype(self.numpy.float32)
		series=series.astype(self.numpy.float32)
		matrix=matrix.astype(self.numpy.float32)
		res=res.astype(self.numpy.float32)
		s = (1 + series.shape[0] + matrix.shape[1] + res.shape[1] + 1)#(len_in.shape[0] + series.shape[0] + matrix.shape[1] + res.shape[0] + deriv.shape[0])#*32 
		#print "[DEBUG] RAM usage: ",s*100.0/n_float, "% (using ",s*4 ," bytes of ",n_float*4, ")"
		self.s = s*100.0/n_float
		if (s*100.0/n_float > 75): return
		#print "[DEBUG] Kernel_time = ", 
		if self.euclidean:
			self.f_euclidean(self.drv.In(len_in), self.drv.In(series), self.drv.In(matrix), self.drv.Out(res), self.drv.In(deriv), block=(self.needed_threads,1,1), grid=(self.blocks,1))#, time_kernel=True)
		else:
			self.f(self.drv.In(len_in), self.drv.In(series), self.drv.In(matrix), self.drv.Out(res), self.drv.In(deriv), block=(self.needed_threads,1,1), grid=(self.blocks,1))#, time_kernel=True)
		#self.drv.Context.synchronize() ##
		return res
	def check_hw(self):
		n = self.drv.Device.count()
		if n == 0: return -1	
		return n
	def __del__(self):
		pass#self.ctx.pop()
	def __init__(self, matrix, derivative = False, euclidean = False):
		self.matrix = matrix
		self.derivative = derivative
		self.euclidean = euclidean
        	#self.drv.init()
		if self.check_hw() == -1: print "Hw not supported."
		self.f_c = self.compiler.SourceModule("""
#include <float.h>

__device__ void deriv_calc(float *a,int l)
{
	int i;
	float t,te;

	t=a[0];
	for (i=1;i<l-1;i+=1)
	{
		te=a[i];
		a[i]=((a[i]-t)+((a[i+1]-t)/2))/2;
		t=te;
	}
	a[0]=a[1]-(a[2]-a[1]);
	a[l-1]=a[l-2]-(a[l-3]-a[l-2]);
}


////// QUESTA E' PER CUDA
__global__ void DTW(float* len_in, float* series, float* matrix, float* res, float* deriv)
{
	//WARNING: il kernel deve essere inizializzati in block mono-dimensionali. 
	//eg: DTW <<< M, K>>>(float* len_in, float* series, float* matrix, float* res, float* deriv)
	//dove: M e' uno scalare, indica il numero di blocks (multicore) su cui viene eseguito il kernel 
	//dove: K e' uno scalare, indica il numero di threads eseguiti per ogni block.

	//DESCRIZIONE INPUT
	//len_in :	lunghezza delle serie in input
	//series: 	puntatore al primo elemento dell'array delle serie 
	//		(tutte le serie sono concatenate, ogni thread lavora sulla coppia corrispondente al proprio idx)
	//		
	//matrix: 	puntatore al blocco di memoria allocato dalla CPU utilizzato per i calcoli per la DTW, 
	//		deve essere len_in*len_in*2*N dove N e' il numero di threads inizializzati 
	//		(tutto lo spazio in un'unica array concatenata, ogni thread lavora sull'area di memoria 
	//		corrispondente al proprio idx)
	//
	//res: 		puntatore all'area in cui vengono scritti i risultati delle DTW, deve essere lungo N.
	//
	//deriv: 	flag che indica se effettuare la derivata sulle serie in imput, 0 per falso, vero altrimenti.
	//

	int len = (int)len_in[0];
	int s_size = len;
	int m_size = len*len;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float *s1 = series+(s_size*idx*2);
	float *s2 = series+((s_size*idx*2)+s_size);
	float *d_matrix = matrix+(m_size*idx*2);
	float *d_values = matrix+((m_size*idx*2)+m_size);

	int i,j;
	if((int)deriv[0]){ deriv_calc(s1, len); deriv_calc(s2, len);}

	//riempio la matrice con le distanze e inizializzo quella per lo short path
	for(i = 0; i < len; i+=1)
	{
		for(j = 0; j < len; j += 1)
		{
			*(d_matrix+(i*len + j)) = sqrt(((i-j)*(i-j)) + ((s1[i] - s2[j])*(s1[i] - s2[j])));
			*(d_values+(i*len + j)) = FLT_MAX;
		}
	}

	//inizializzo il primo elemento
	*d_values = *d_matrix;

	// riempio la matrice per lo short path eccetto l'ultima riga e l'ultima colonna
	for(i = 0; i < len-1; i+=1)
	{
		for(j = 0; j < len-1; j += 1)
		{
			*(d_values+((i+1)*len + j+1)) = *(d_values+(i*len + j)) + *(d_matrix+((i+1)*len + j + 1));
			if(*(d_values+(i*len + j)) + *(d_matrix+(i*len + j+1)) < *(d_values+(i*len + j+1))) 
				*(d_values+(i*len + j+1)) = *(d_values+(i*len + j)) + *(d_matrix+(i*len + j+1));
			if(*(d_values+(i*len + j)) + *(d_matrix+((i+1)*len + j)) < *(d_values+((i+1)*len + j))) 
				*(d_values+((i+1)*len + j)) = *(d_values+(i*len + j)) + *(d_matrix+((i+1)*len + j));
		}
	}
	//riempio l'ultima riga e l'ultima colonna (senza l'ultima cella)
	i = len-1;
	for (j = 0; j < len-1; j +=1)
	{
		if(*(d_values+(i*len + j)) + *(d_matrix+(i*len + j+1)) < *(d_values+(i*len + j+1))) 
			*(d_values+(i*len + j+1)) = *(d_values+(i*len + j)) + *(d_matrix+(i*len + j+1));
	}
	j = len-1;
	for (i = 0; i < len-1; i +=1)
	{
		if(*(d_values+(i*len + j)) + *(d_matrix+((i+1)*len + j)) < *(d_values+((i+1)*len + j)))
			*(d_values+((i+1)*len + j)) = *(d_values+(i*len + j)) + *(d_matrix+((i+1)*len + j));
	}
	// i e j sono len-1, metto a posto l'ultima cella
	if (*(d_values+((i-1)*len + j)) + *(d_matrix+(i*len + j)) < *(d_values+(i*len + j)))
		*(d_values+(i*len + j)) = *(d_values+((i-1)*len + j)) + *(d_matrix+(i*len + j));
	if (*(d_values+((i-1)*len + j-1)) + *(d_matrix+(i*len + j)) < *(d_values+(i*len + j)))
		*(d_values+(i*len + j)) = *(d_values+((i-1)*len + j-1)) + *(d_matrix+(i*len + j));
	if (*(d_values+((i)*len + j-1)) + *(d_matrix+(i*len + j)) < *(d_values+(i*len + j)))
		*(d_values+(i*len + j)) = *(d_values+(i*len + j-1)) + *(d_matrix+(i*len + j));

	res[idx] = *(d_values+(i*len + j));
}		


//USA POCA RAM
__global__ void compute_dtw(float* len_in, float* series, float* matrix, float* res, float* deriv)
{
	//WARNING: il kernel deve essere inizializzati in block mono-dimensionali. 
	//eg: DTW <<< M, K>>>(float* len_in, float* series, float* matrix, float* res, float* deriv)
	//dove: M e' uno scalare, indica il numero di blocks (multicore) su cui viene eseguito il kernel 
	//dove: K e' uno scalare, indica il numero di threads eseguiti per ogni block.

	//DESCRIZIONE INPUT
	//len_in :	lunghezza delle serie in input
	//series: 	puntatore al primo elemento dell'array delle serie, deve essere 2*len_in*N 
	//		dove N e' il numero di threads inizializzati 
	//		(tutte le serie sono concatenate, ogni thread lavora sulla coppia corrispondente al proprio idx)
	//		
	//matrix: 	puntatore al blocco di memoria allocato dalla CPU utilizzato per i calcoli per la DTW, 
	//		deve essere 4*len_in*N.
	//		(tutto lo spazio in un'unica array concatenata, ogni thread lavora sull'area di memoria 
	//		corrispondente al proprio idx)
	//
	//res: 		puntatore all'area in cui vengono scritti i risultati delle DTW, deve essere lungo N.
	//
	//deriv: 	flag che indica se effettuare la derivata sulle serie in imput, 0 per falso, vero altrimenti.
	//

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int len = (int)len_in[0];
	float *s1 = series+(len*idx*2);
	float *s2 = series+((len*idx*2)+len);
	float *d_values_l = matrix+(len*idx*4);
	float *d_values_u = matrix+((len*idx*4)+len);
	float *p_values_l = matrix+((len*idx*4)+2*len);
	float *p_values_u = matrix+((len*idx*4)+3*len);
	float *ex;
	int i,j;
	if((int)deriv[0]){ deriv_calc(s1, len); deriv_calc(s2, len);}

	//inizializzo il primo elemento

	for (j = 0; j < len; j++)
		d_values_l[j] = ((0-j)*(0-j)) + ((s1[0] - s2[j])*(s1[0] - s2[j]));
	p_values_l[0] = d_values_l[0];
	for (j = 0; j < len-1; j++)
		p_values_l[j+1] = p_values_l[j] + d_values_l[j+1];
	
	for (i = 0; i < len-1; i++)
	{	
		for (j = 0; j < len; j++)
			d_values_u[j] = (((i+1)-j)*((i+1)-j)) + ((s1[i+1] - s2[j])*(s1[i+1] - s2[j]));

		p_values_u[0] = FLT_MAX;	
		
		for (j = 0; j < len-1; j++)
		{
			p_values_u[j+1] = d_values_u[j+1] + p_values_l[j];
			if (p_values_l[j] + d_values_l[j+1] < p_values_l[j+1]) p_values_l[j+1] = p_values_l[j] + d_values_l[j+1];
			if (p_values_l[j] + d_values_u[j] < p_values_u[j]) p_values_u[j] = p_values_l[j] + d_values_u[j];
		}
	
		if (p_values_l[j] + d_values_u[j] < p_values_u[j]) p_values_u[j] = p_values_l[j] + d_values_u[j];
		ex = d_values_u;
		d_values_u = d_values_l;
		d_values_l = ex;
		ex = p_values_u;
		p_values_u = p_values_l;
		p_values_l = ex;
	}

	for (j = 0; j < len-1; j++)
	{
		if (p_values_l[j] + d_values_l[j+1] < p_values_l[j+1]) p_values_l[j+1] = p_values_l[j] + d_values_l[j+1];
	}
	
	res[idx] = p_values_l[j];
}		

__global__ void EUCLIDEAN(float* len_in, float* series, float* res, float* deriv)
{
	//WARNING: il kernel deve essere inizializzati in block mono-dimensionali. 
	//eg: DTW <<< M, K>>>(float* len_in, float* series, float* matrix, float* res, float* deriv)
	//dove: M e' uno scalare, indica il numero di blocks (multicore) su cui viene eseguito il kernel 
	//dove: K e' uno scalare, indica il numero di threads eseguiti per ogni block.

	//DESCRIZIONE INPUT
	//len_in :	lunghezza delle serie in input
	//series: 	puntatore al primo elemento dell'array delle serie, deve essere 2*len_in*N 
	//		dove N e' il numero di threads inizializzati 
	//		(tutte le serie sono concatenate, ogni thread lavora sulla coppia corrispondente al proprio idx)
	//
	//res: 		puntatore all'area in cui vengono scritti i risultati delle DTW, deve essere lungo N.
	//
	//deriv: 	flag che indica se effettuare la derivata sulle serie in imput, 0 per falso, vero altrimenti.
	//

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int len = (int)len_in[0];
	float *s1 = series+(len*idx*2);
	float *s2 = series+((len*idx*2)+len);
	float sum = 0;
	int j;
	if((int)deriv[0]){ deriv_calc(s1, len); deriv_calc(s2, len);}
	
	for (j = 0; j < len; j++)
		sum += (s1[j] - s2[j])*(s1[j] - s2[j]);
	res[idx] = sum;
}		





///////QUESTA CALCOLA ANCHE IL PATH (NON PER CUDA)
__global__ void DTW_PATH(float* len_in, float* series, float* matrix, float* res, float* deriv,float* path)
{
	//i kernel devono essere inizializzati in block mono-dimensionali. 
	//eg: DTW <<< M, N>>>(lung_serie_d, serie_d, b_d, res_d, deriv_d, path_d); 
	//dove: M e' uno scalare, indica il numero di blocks (multicore) su cui viene eseguito il kernel 
	//dove: N e' uno scalare, indica il numero di threads eseguiti per ogni block.
	int len = (int)len_in[0];
	int s_size = len;
	int m_size = len*len;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float *s1 = series+(s_size*idx*2);
	float *s2 = series+((s_size*idx*2)+s_size);
	float *d_matrix = matrix+(m_size*idx*2);
	float *d_values = matrix+((m_size*idx*2)+m_size);

	int i,j,n;
	if((int)deriv[0]){ deriv_calc(s1, len); deriv_calc(s2, len);}

	//riempio la matrice con le distanze e inizializzo quella per lo short path
	for(i = 0; i < len; i+=1)
	{
		for(j = 0; j < len; j += 1)
		{
			*(d_matrix+(i*len + j)) = sqrt(((i-j)*(i-j)) + ((s1[i] - s2[j])*(s1[i] - s2[j])));
			*(d_values+(i*len + j)) = FLT_MAX;
		}
	}

	//inizializzo il primo elemento
	*d_values = *d_matrix;

	// riempio la matrice per lo short path eccetto l'ultima riga e l'ultima colonna
	for(i = 0; i < len-1; i+=1)
	{
		for(j = 0; j < len-1; j += 1)
		{
			*(d_values+((i+1)*len + j+1)) = *(d_values+(i*len + j)) + *(d_matrix+((i+1)*len + j + 1));
			if(*(d_values+(i*len + j)) + *(d_matrix+(i*len + j+1)) < *(d_values+(i*len + j+1))) 
				*(d_values+(i*len + j+1)) = *(d_values+(i*len + j)) + *(d_matrix+(i*len + j+1));
			if(*(d_values+(i*len + j)) + *(d_matrix+((i+1)*len + j)) < *(d_values+((i+1)*len + j))) 
				*(d_values+((i+1)*len + j)) = *(d_values+(i*len + j)) + *(d_matrix+((i+1)*len + j));
		}
	}
	//riempio l'ultima riga e l'ultima colonna (senza l'ultima cella)
	i = len-1;
	for (j = 0; j < len-1; j +=1)
	{
		if(*(d_values+(i*len + j)) + *(d_matrix+(i*len + j+1)) < *(d_values+(i*len + j+1))) 
			*(d_values+(i*len + j+1)) = *(d_values+(i*len + j)) + *(d_matrix+(i*len + j+1));
	}
	j = len-1;
	for (i = 0; i < len-1; i +=1)
	{
		if(*(d_values+(i*len + j)) + *(d_matrix+((i+1)*len + j)) < *(d_values+((i+1)*len + j)))
			*(d_values+((i+1)*len + j)) = *(d_values+(i*len + j)) + *(d_matrix+((i+1)*len + j));
	}
	// i e j sono len-1, metto a posto l'ultima cella
	if (*(d_values+((i-1)*len + j)) + *(d_matrix+(i*len + j)) < *(d_values+(i*len + j)))
		*(d_values+(i*len + j)) = *(d_values+((i-1)*len + j)) + *(d_matrix+(i*len + j));
	if (*(d_values+((i-1)*len + j-1)) + *(d_matrix+(i*len + j)) < *(d_values+(i*len + j)))
		*(d_values+(i*len + j)) = *(d_values+((i-1)*len + j-1)) + *(d_matrix+(i*len + j));
	if (*(d_values+((i)*len + j-1)) + *(d_matrix+(i*len + j)) < *(d_values+(i*len + j)))
		*(d_values+(i*len + j)) = *(d_values+(i*len + j-1)) + *(d_matrix+(i*len + j));

	res[idx] = *(d_values+(i*len + j));

	i = len-1;
	j = len-1;
	n = 0;
	while(1)
	{
		path[n] = i;
		path[n+1] = j;
		n += 2;
		if ((j == 0) && (i == 0)) break;
		if (i == 0) {j -= 1; continue;}
		if (j == 0) {i -= 1; continue;}
		if (d_values[(i-1)*len+j] < d_values[(i-1)*len+j-1] && d_values[(i-1)*len+j] < d_values[i*len+j-1]) {i -= 1; continue;}
		if (d_values[(i-1)*len+j-1] < d_values[i*len+j-1] && d_values[(i-1)*len+j-1] < d_values[(i-1)*len+j]) {j -= 1; i -= 1; continue;}
		j -= 1;
	}
	path[n] = -4;
	path[n+1] = -4;
}	


""");
