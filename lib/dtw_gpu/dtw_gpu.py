#!/usr/bin/env python

class GpuDistance(object):
    try:
        import pycuda.driver as driver
        import pycuda.compiler as compiler
        import pycuda.autoinit
    except:
        raise ImportError("Cannot find PyCUDA")
    import numpy    
    
    def hwCheck(self):
        n = self.driver.Device.count()
        if n == 0: return -1	
        return n
	
    def printFeatures(self, device):
        print "SM count: ", self.driver.Device(device).multiprocessor_count
        print "Max shared memory per block: ", self.driver.Device(device).max_shared_memory_per_block
        print "Max registers per block: ", self.driver.Device(device).max_registers_per_block
        print "Max threads per block: ", self.driver.Device(device).max_threads_per_block
        print "Max block x: ", self.driver.Device(device).max_block_dim_x					
        print "Max block y: ", self.driver.Device(device).max_block_dim_y
        print "Max block z: ", self.driver.Device(device).max_block_dim_z
        print "Max gird x: ", self.driver.Device(device).max_grid_dim_x
        print "Max gird y: ", self.driver.Device(device).max_grid_dim_y
        print "Max gird z: ", self.driver.Device(device).max_grid_dim_z

    def launch(self, dtwlist_in):
        if self.method == "dtw" or self.method == "ddtw":
            function = self.source.get_function('calc_dtw')
        elif self.method == "euclidean":
            function = self.source.get_function('euclidean')
        
        dtwlist = self.numpy.array(dtwlist_in)
        dtwlist = dtwlist.astype(self.numpy.float32)
        dtwnum = dtwlist.__len__()       
        results = self.numpy.empty(dtwnum)
        results = results.astype(self.numpy.float32)
        param = self.numpy.array(self.len)
        param = param.astype(self.numpy.float32)
        
        if self.len < 512:
            threadsPerBlock = self.len
        else:
            threadsPerBlock = 512
   
        result_total = self.numpy.empty(0)
        cont = 65535
        while cont < dtwnum:     
           function(self.matrix_gpu, self.driver.In(dtwlist),self.driver.In(param), self.driver.Out(results), block = (threadsPerBlock,1,1), grid = (65535,1) )
           self.numpy.concatenate((result_total, results))
           cont += 65535
        function(self.matrix_gpu, self.driver.In(dtwlist),self.driver.In(param), self.driver.Out(results), block = (threadsPerBlock,1,1), grid = (dtwnum%cont,1) )

        result_total = self.numpy.concatenate((result_total, results))
        return result_total

    def calc_deriv(self):
        function = self.source.get_function('calc_deriv')
        num = self.numpy.array(self.matrix.__len__())
        num = num.astype(self.numpy.float32)
        lenght = self.numpy.array(self.len)
        lenght = lenght.astype(self.numpy.float32)

        if self.len < 512:
            threadsPerBlock = self.len
        else:
            threadsPerBlock = 512
        
        cont = 65535
        while cont < num:
            function(self.matrix_gpu, self.driver.In(num) , self.driver.In(lenght), block = (threadsPerBlock,1,1), grid = (65535,1))
            cont += 65535            
        function(self.matrix_gpu, self.driver.In(num) , self.driver.In(lenght), block = (threadsPerBlock,1,1), grid = (self.matrix.__len__()%cont,1))
        
    def __init__(self, matrix, mode = "dtw" , deriv = True):
        self.matrix = self.numpy.array(matrix)
        self.matrix = self.matrix.astype(self.numpy.float32)
        self.matrix_gpu = self.driver.mem_alloc(self.matrix.nbytes)
        self.driver.memcpy_htod(self.matrix_gpu, self.matrix)
        self.len = self.matrix[0].__len__()
        self.method = mode
        self.source = self.compiler.SourceModule("""
#define MAX_THREADS_PER_BLOCK 512

__global__ void euclidean(float* series, float* ope, float* param, float *results)
{
    int len = param[0];
    int i,j;
    float* s1 = series+len*((int)ope[blockIdx.x*2]);
    float* s2 = series+len*((int)ope[blockIdx.x*2+1]);
    __shared__ float sum[MAX_THREADS_PER_BLOCK];  

    if (threadIdx.x == 0) results[blockIdx.x] = 0;
    __syncthreads();

    for(i = threadIdx.x; i < len; i += MAX_THREADS_PER_BLOCK)
    {
        sum[i] = (s1[i] - s2[i])*(s1[i] - s2[i]);
        __syncthreads();
        if (threadIdx.x == 0)
        {
             for(j = 0; (i+MAX_THREADS_PER_BLOCK < len ? j < MAX_THREADS_PER_BLOCK : j < len%MAX_THREADS_PER_BLOCK); j++)
                results[blockIdx.x] += sum[j];
        }
    }
}

__global__ void calc_deriv(float* series, float *num, float* len)
{
    float* s = series + (int)(len[0]*blockIdx.x);
    int i;
    float val1, val2;
    
    for(i = threadIdx.x; i < (int)len[0]-1; i += MAX_THREADS_PER_BLOCK)
    {
        val1 = s[i];
        val2 = s[i+1];
        __syncthreads(); //first all threads have to read the values, and then they can proceed and write the results.
        s[i] = val1 - val2;
    }
    if (i == (int)len[0]-1) //only one thread
        s[i] = s[i-1];
}

__global__ void calc_dtw(float* series, float* dtws, float* param, float *results)
{

    int len = param[0];
    int i,j;
    //alloc d_val and p_val in the shared scope
    /*instead of allocating the whole distance and path matrixs, we only use 2 rows of each at a time. Hence, the memory usage is linear and 
    shared memory can be used. I.E. d_val_l is the lover row of the distance matrix, p_val_u is the upper one in the path matrix. 
    We iterate through the matrix continously updating these two rows, at the end of each iteration the upper rows becomes the lover ones, 
    and new upper rows are initialized.*/
    __shared__ float d_val_u_main[MAX_THREADS_PER_BLOCK];
    __shared__ float d_val_l_main[MAX_THREADS_PER_BLOCK];
    __shared__ float p_val_u_main[MAX_THREADS_PER_BLOCK];
    __shared__ float p_val_l_main[MAX_THREADS_PER_BLOCK];
    float* d_val_u = d_val_u_main;
    float* d_val_l = d_val_l_main;
    float* p_val_u = p_val_u_main;
    float* p_val_l = p_val_l_main;
    float* ex;    
    float* s1 = series+len*((int)dtws[blockIdx.x*2]);
    float* s2 = series+len*((int)dtws[blockIdx.x*2+1]);
    
    //create distance table initializing d_val_l
    d_val_l[threadIdx.x] = ((0-threadIdx.x)*(0-threadIdx.x)) + ((s1[0] - s2[threadIdx.x])*(s1[0] - s2[threadIdx.x]));

    if (threadIdx.x == 0) //this part is not parallelizable, so only one thread per block does all the work =(
    {
        	p_val_l[0] = d_val_l[0]; //initialize p_val_l
	        for (i = 0; i < len-1; i++)
		        p_val_l[i+1] = p_val_l[i] + d_val_l[i+1];
    }        
    __syncthreads();
    
    //here comes the fun
    for (i = 1; i < len; i++) //let's go through all the matrix
    {
        d_val_u[threadIdx.x] = ((i-threadIdx.x)*(i-threadIdx.x)) + ((s1[i] - s2[threadIdx.x])*(s1[i] - s2[threadIdx.x])); // initialize d_val_u
       
        p_val_u[threadIdx.x] = p_val_l[threadIdx.x] + d_val_u[threadIdx.x];        //search shortest path
        __syncthreads();
        
        if((p_val_u[threadIdx.x+1] > d_val_u[threadIdx.x+1] + p_val_l[threadIdx.x]) && threadIdx.x != MAX_THREADS_PER_BLOCK  && threadIdx.x < len-1) 
            p_val_u[threadIdx.x+1] = d_val_u[threadIdx.x+1] + p_val_l[threadIdx.x];    
        __syncthreads();
        
        for(j = threadIdx.x; j != MAX_THREADS_PER_BLOCK && j < len-1; j++) 
        {
            if (p_val_u[j] + d_val_u[j+1] < p_val_u[j+1])
            {
                //__syncthreads();
                p_val_u[j+1] = p_val_u[j] + d_val_u[j+1];
            }
            else break;
        }
        __syncthreads();
        
        ex = d_val_u;    //exchanging pointers....
		d_val_u = d_val_l;
		d_val_l = ex;
		ex = p_val_u;
		p_val_u = p_val_l;
		p_val_l = ex;
    }
    __syncthreads();
    
    //we're done! =) Only remains to return the result!
    if (threadIdx.x == 0) //last operation.... only one thread works
    {
        results[blockIdx.x] = p_val_l[len-1];
    }
}
""");
        if deriv:
            self.calc_deriv()
