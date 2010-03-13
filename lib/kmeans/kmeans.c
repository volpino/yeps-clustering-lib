#include <stdio.h>
#include <stdbool.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <Python.h>

int *nrip,*seed;
char *distance;
char *pu;
bool fast;
double radius,error;

void init (int *, char *, bool, int, int *, double,char *);
void compute(int, double **, int , int );
void select_centroids(int , int, int *, double **);
void compare(int, double **, int , int, int *, double **);
void compare_cpu(int , double **, int , int , int *, double **);
double difference (int , double *,double *);
double difference_dtw(double *, double *);
double difference_eucl(int ,double *, double *);
double difference_pearson(double *, double *);
void compare_gpu(int , int , int *, double **);
void control(int , double **, int, int , int *, double **);
double calc_err(int ,int , int *, double **);
void newcentroids(int , double **, int , int , int *, double **);



void compute(int k, double **mat, int r, int c, double **outp_centr, int *outp_series)
{
	int i,j;
	int centroids[k],cont;
	double old_error, new_error;
	double *min[r];
	double centroids_outp[k][c];
	for(i = 0; i < r; i+=1)
		min[i] = (double*) malloc(sizeof(double)*2);

	for (i=0;i<k;i+=1)
		centroids[i]=-1;
	if (seed==NULL)
		srand(time(NULL));
	else
		srand(*seed);
	select_centroids(k,r,centroids,min);
	compare(k,mat,r,c,centroids,min);
	control(k,mat,r,c,centroids,min);
	old_error=calc_err(k,r,centroids,min);
	new_error = old_error*(2+error);
	cont=0;
	if (nrip==NULL)
	{
		while ((abs(new_error/old_error-1)>error) && (cont<500))
		{
			newcentroids(k,mat,r,c,centroids,min);
			compare(k,mat,r,c,centroids,min);
			cont+=1;
			old_error=new_error;
			new_error=calc_err(k,r,centroids,min);
		}
	}
	else
		for (i=0;i<*nrip;i++)
		{
			newcentroids(k,mat,r,c,centroids,min);
			compare(k,mat,r,c,centroids,min);
		}
	for (i=0;i<k;i+=1)
		for (j=0;j<c;j+=1)
			centroids_outp[i][j]=mat[i+r][j];
	for (i=0;i<r;i+=1)
		series_outp[i]=min[i][0]-r+1;
	for(i = 0; i < r; i+=1)
		free(min[i]);
}

void select_centroids(int k, int r, int *centroids, double **min)
{
	int i,j,cond,t;


	for (i=0;i<k;i+=1)
	{
		cond = 0;
		while (cond==0)
		{
			t=rand()%r;
			for (j=0;j<k;j+=1)
				if (centroids[i]==t)
					cond=0;
				else
				{
					centroids[i]=t;
					cond = 1;
				}
		}
	}
}

void compare(int k, double **mat, int r, int c, int *centroids, double **min)
{
	if (strcmp(pu,"GPU")==0)
		compare_gpu(k,r,centroids,min);
	else
		compare_cpu(k,mat,r,c,centroids,min);
}

void compare_cpu(int k, double **mat, int r, int c, int *centroids, double **min)
{
	int i,j,z;
	double val;
	double listdiff[k];
	double t[c], te[c];	

	for (i=0; i<r; i+=1)
	{
		for (j=0; j<k; j+=1)
			listdiff[j]=0;
		for (j=0; j<k; j++)
		{
			for (z=0;z<c;z+=1)
				t[z]=mat[i][z];
			for (z=0;z<c;z+=1)
				te[z]=mat[j][z];
			listdiff[j]= difference(c,t,te);
		}
		val=listdiff[0];
		min[i][0]=centroids[0];
		min[i][1]=val;
		for (j=0; j<k; j+=1)
			if (val>listdiff[j])
			{
				val=listdiff[j];
				min[i][0]=centroids[j];
				min[i][1]=val;
			}	
	}
}

double difference (int c, double *a,double *b)
{
	if ( (strcmp(distance,"dtw")==0) || (strcmp(distance,"ddtw")==0))
		return difference_dtw (a, b);
	if (strcmp(distance,"euclidean")==0)
		return difference_eucl (c, a, b);
	if ((strcmp(distance,"pearson")==0))
		return difference_pearson (a, b);
}

double difference_dtw(double *a, double*b)
{
	bool derivative;

	if (strcmp(distance,"ddtw")==0)
		derivative=true;
	else
		derivative=false;
	return 0.0001;//dtw_cpu.compute_dtw(a, b, false, false, derivative, fast, radius);
}

double difference_eucl(int c,double *a, double*b)
{
	int i;
	double val;

	val=0;
	for (i=0; i<c; i+=1)
		val+= pow((a[i]-b[i]),2);
	return val;
}

double difference_pearson(double *a, double*b)
{
	//return scipy.stats.pearsonr(a,b);
}

void compare_gpu(int k, int r, int *centroids, double **min)
{
	int i,j;
	int li[i*k][2];
	double lista[i*k];
	double minimum;

	
	for (i=0; i<r; i+=1)
		for (j=0; j< k; j+=1)
		{
			li[i*j+j][0]=i;
			li[i*j+j][1]=centroids[j];
		}
	//lista=l.compute(&li);
	
	for (i=0; i<r; i+=1)
	{
		minimum=lista[i*k+0];
		min[i][0]=centroids[0];
		min[i][1]=lista[i*k+0];
		for (j=0; j<k; j+=1)
			if (minimum>=lista[i*k+j])
			{
				min[i][0]=centroids[j];
				min[i][1]=lista[i*k+j];
				minimum=lista[i*k+j];
			}
	}
}

void control(int k, double **mat, int r, int c, int *centroids, double **min)
{
	int i;
	double cond[r];
	bool check;

	for (i=0; i<r; i+=1)
		cond[i]=0;
	for (i=0; i<r; i+=1)
		cond[(int)min[i][0]]+=1;
	check=false;
	for (i=0; i<r; i+=1)
		if (cond[i]==1)
			check==true;
	if (check)
	{
		printf ("EMPTY CLUSTER:RE-INIT");
		select_centroids(k,r,centroids,min);
		compare(k,mat,r,c,centroids,min);
		control(k,mat,r,c,centroids,min);
	}
}

double calc_err(int k, int r, int *centroids, double **min)
{
	int i,j;
	double varianza[k];
	double sum;	
	int div;

	for (i=0; i<k;i+=1)
		varianza[i]=0;
	for (i=0; i<k; i+=1)
	{
		div=0;
		for (j=0; j<r; j+=1)
			if (min[j][0]==centroids[i])
				varianza[i]+=pow(min[j][1],2);
				div+=1;
		varianza[i]/=div;
	}
	sum=0;
	for (i=0; i<k;i+=1)
		sum+=varianza[i];
	return sum;
}

void newcentroids(int k, double **mat, int r, int c, int *centroids, double **min)
{
	int i,j,z;
	double media[c];
	int index;


	for (i=0; i<k; i++)
	{
		for (j=0;j<c; j+=1)
			media[j]=0;
		index=0;
		for (j=0; j<r; j+=1)
			if (min[j][0]==centroids[i])
			{
				index+=1;
				for (z=0;z<c; z+=1)
					media[z]+=mat[j][z];
			}
		for (j=0; j<c; j+=1)
			media[j]/=index;
		mat[r+i]=media;
		centroids[i]=r+i;
	}
}

int main()
{
	int i;
	double **centroids;
	double **min;
	int k = 2;
	double *mat[9]; //create and alloc variables
	double a[9][5]=
{{4,2,1,6,1},{4,2,3,6,6},{1,2,1,4,4},{7,5,7,4,1},{8,6,5,7,2},{7,8,1,9,1},{1,2,4,4,3},{7,5,9,7,4},{3,1,2,4,5}};
	for (i=0;i<9;i++)
		mat[i]=a[i];

	init (NULL, "ddtw", false, 20, NULL, 0.0001, "CPU");
	compute (k, mat, 9, 5);//, centroids, min);
    
	return 0;
}


static PyObject *
kmeans_init(PyObject *self, PyObject *args)
{
	int it, s;
	char *dist, *pr_u;
	bool fast_dtw,
	double tol, rad;
	if (!PyArg_ParseTuple(args, "i s i d i d s", &it, &dist, &fast_dtw, &rad, &s, &tol, &pr_u))
		return NULL;
	nrip=&it;
	distance=dist;
	fast=fast_dtw;
	radius=rad;
	error=tol;
	pu=pr_u;
	seed=&s;
}

static PyObject *
kmeans_compute(PyObject *self, PyObject *args)
{
	PyObject *mat = NULL;
	PyObject *matc = NULL;
	PyObject *centroids = NULL;
	PyObject *series = NULL;
	double *adata, **centroids_data;
	int k,matndim, r, c,centroids_dim[2], *series_data;

	if (!PyArg_ParseTuple(args, "i O", &k, &mat))
		return NULL;

	matc = PyArray_FROM_OTF(mat, NPY_DOUBLE, NPY_IN_ARRAY);
	matdata = (double *) PyArray_DATA(matc);
	matndim = PyArray_NDIM(matc);

	if ((matndim != 2)
	{
		PyErr_SetString(PyExc_ValueError, "a and b should be 1D array");
		return NULL;
	}

	r = PyArray_DIM(matc, 0);
	c = PyArray_DIM(matc, 1);

	centroids_dim[0]=k;
	centroids_dim[1]=r;	
	centroids = PyArray_SimpleNew(2,centroids_dim,NPY_DOUBLE);
	centroids_data = (double *) PyArray_DATA(centroids);	
	
	series_dim=r;
	series = PyArray_SimpleNew(1,series_dim,NPY_INT);
	series_data = (int *) PyArray_DATA(series);

	compute(k, matdata, r, c, centroids_data, series_data);

	return Py_BuildValue("N, N", centroids, series);
}





static PyMethodDef KmeansMethods[] = {
    { "kmeans.init",kmeans_init,
      METH_VARARGS},
    { "kmeans.compute", kmeans_compute,
      METH_VARARGS},
    {NULL, NULL, 0, NULL} /* Sentinel */
}


PyMODINIT_FUNC
initkmeans(void)
{
   (void)Py_InitModule("kmeans", KmeansMethods);
   import_array();
}




