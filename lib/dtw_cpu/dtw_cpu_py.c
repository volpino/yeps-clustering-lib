#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#include <Python.h>
#include <numpy/arrayobject.h>

/* C implementation of the DTW/FastDTW algorithm, usin a recursive function to calculate the min distance.
 *The recursive function starts from the left bottom edge of the matrix and calculates recursively the min path to reach the top right edge. For each
 *cell it stores the minimum path returned by adiacent cell. The calculation stops when the top right edge is reached by all of the recoursion
 *launched.
 *
 *The main func is totally useless, only the DTW func should be used. The parameters should be passed in this order:
 *(double* frist_time_series, int frist_time_series_len, double* second_time_series, int frist_time_series_len, int radius, point_t* path, int deri
 *Radius is a parameter for the FastDTW algorithm, pass -1 to use the classic DTW, only positive (or zero) value are allowed (execpt for -1)
 *increasing radius means more accurate result, but may decrease (even significantly) the speed.
 *If path is NULL, the path isn't returned, else the path will be written into path pointer; path should be already allocated, the maximum lenght of
 *path is len_frist_series + len_second_seies +1 . The end of the path array is signaled by x and y values set to -1.
 *If deriv is 1 the calculation is made on the derivatives of the two series.
 */

typedef struct point_t // struct used for the storage of the path
{
	int x,y;
} point_t;

double min(double a, double b, double c) //search the minimum (this comment is pretty useless XD)
{
	if (a>b)
		a=b;
	if (a>c)
		a=c;
	return a;
}
void process_input(double* a1, double* a6, int a2, int a3, int a4, int a5);
void deriv_calc(double *a,int l)
{
	if (l < 3)return;
	int i;
	double t,te;

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

void fill_matrix(double **matrix, double* s1, int len_s1, double* s2, int len_s2) //fill the distance matrix
{
	int i,j;  // counters
	for(i = 0; i < len_s1; i+=1) //iteration throug the x axis
	{
		for(j = 0; j < len_s2; j+=1) //iteration throug the y axis
		{
			matrix[i][j] = pow((i - j),2) + pow((s1[i] - s2[j]),2); //distance calculation
		}
	}
}

double short_path(double **matrix, int len_x, int len_y, point_t* path) //calculates the shortest path
{
	//double d_values[len_x][len_y];  //min path matrix
	double **d_values = malloc(sizeof(double*) * len_x);
	int i,j,n; //counters
	double res;
	double next(int x, int y)   //recursive function that goes throug the distance matrix and stores into the min matrix the minumum distance from each cell to the end of the matrix.
	{
		double a,b,c;
		if (x == len_x-1 && y == len_y -1) return matrix[x][y]; //if we've reached the end, return;
		if (x >= len_x || y >= len_y || matrix[x][y] == -1) return FLT_MAX; //if we're out of the matrix return
		if (d_values[x][y] != -1) return d_values[x][y]; //if the value for this cell has already been calculated, return
		a = next(x,y+1); //calculate values for adiacent cells
		b = next(x+1, y+1);
		c = next(x+1,y);
		d_values[x][y] = min(a,b,c) + matrix[x][y]; //calculate the value for the curren cell
		return d_values[x][y];
	}

	for (i=0; i < len_x; i++)
		d_values[i] = malloc(sizeof(double)*len_y);

	for (i = 0; i < len_x; i += 1) //initialize the min matrix to -1 (-1 means not calculated yet)
	{
		for (j = 0; j < len_y; j +=1)
		{
			d_values[i][j] = -1;
		}
	}
	res = next(0,0); // run recursive func

	if (path == NULL)
	{
			for (i=0; i < len_x; i++) free(d_values[i]);
			free(d_values);
			return res; //if we haven't to calculate the path return
	}


	i = 0; //initialize conunters
	j = 0;
	n = 0;
	while(1) //this goes throug the matrix searching the path
	{
		path[n].x = i; //add the current point to the path array
		path[n].y = j;
		n +=1; //increment array counter
		if(( i == len_x -1) && (j == len_y-1 )) break; //if we've reached the end, stop;
		if (i == len_x-1) {j += 1;continue;} //if we are on the matrix limit, there's only one way.
		if (j == len_y-1) {i += 1;continue;}
		if (d_values[i+1][j+1] < d_values[i][j+1] && d_values[i+1][j+1] < d_values[i+1][j]) {j += 1; i += 1; continue;} //find the min values in the near cells and update coordinates
		if (d_values[i+1][j] < d_values[i+1][j+1] && d_values[i+1][j] < d_values[i][j+1]) {i += 1; continue;}
		j += 1;
	}
	path[n].x = -1; //this is to signal the end of the path array
	path[n].y = -1;
	for (i=0; i < len_x; i++) free(d_values[i]);
	free(d_values);
	return res;
}


double DTW(double* s1, int len_s1, double* s2, int len_s2, point_t* path)
{
	double *mat[len_s1]; //create and alloc variables
	double res;
	int i;
	for(i = 0; i < len_s1; i+=1) mat[i] = (double*) malloc(sizeof(double)*len_s2);

	fill_matrix(mat, s1, len_s1, s2, len_s2); //fill the matrix

	res = short_path(mat, len_s1,len_s2, path); //calculate shortest path
	for(i = 0; i < len_s1; i+=1) free(mat[i]);  //free useless memory
	return res; //return
}

//*****************************************************************************************************************************************************

void HalfRes(double* a, int len, double* result)
{
	int i;
	for(i = 0; i < (len/2); i += 1)
		result[i] = a[2*i] + a[(2*i)+1];
	if (len % 2 == 1) result[(len/2)+1] = a[len];
}

void fast_fill_matrix(double **matrix, point_t *window, int radius, double* s1, int len_s1, double* s2, int len_s2)
{
	int i, x,y;
	for (x = 0; x < len_s1; x += 1) //initialize the min matrix to -1 (-1 means not calculated yet)
	{
		for (y = 0; y < len_s2; y +=1)
		{
			matrix[x][y] = -1;
		}
	}

	for(i = 0; window[i].x != -1; i +=1)
	{
		for (x = (window[i].x*2)-radius-2; x <=(window[i].x*2)+radius+1; x += 1)
		{
			for(y = (window[i].y*2)-radius-2; y <= (window[i].y*2)+radius+1; y +=1)
			{
				if ((x < len_s1 && x > -1) && (y < len_s2 && y > -1))
					matrix[x][y] = pow((x - y),2) + pow((s1[x] - s2[y]),2);
			}
		}
	}
}

double calc_FastDTW(double* s1, int len_s1, double* s2, int len_s2, point_t* window, int radius, point_t* path)
{
	double *mat[len_s1]; //create and alloc variables
	double res;
	int i;
	for(i = 0; i < len_s1; i+=1) mat[i] = (double*) malloc(sizeof(double)*len_s2);
	fast_fill_matrix(mat, window, radius, s1, len_s1, s2, len_s2); //fill the matrix
	res = short_path(mat, len_s1,len_s2, path); //calculate shortest path
	for(i = 0; i < len_s1; i+=1) free(mat[i]);  //free useless memory
	return res; //return
}


double FastDTW(double* s1, int len_s1, double* s2, int len_s2, int radius, point_t* path)
{
	int minTsize = (radius*3)+2;
	int shrunkS1_len = len_s1/2 + (len_s1 % 2);
	int shrunkS2_len = len_s2/2 + (len_s2 % 2);
	double shrunkS1[shrunkS1_len];
	double shrunkS2[shrunkS2_len];
	point_t lowResPath[shrunkS1_len + shrunkS2_len];
	if(len_s1 < minTsize || len_s2 < minTsize)
	{
		return DTW(s1, len_s1, s2, len_s2, path);
	}

	HalfRes(s1, len_s1, shrunkS1);
	HalfRes(s2, len_s2, shrunkS2);
	FastDTW(shrunkS1, shrunkS1_len, shrunkS2, shrunkS2_len, radius, lowResPath);
	return calc_FastDTW(s1, len_s1, s2, len_s2, lowResPath, radius, path);
}

double euclidean(double* s1, int len_s1, double* s2, int len_s2)
{
	int i;
	double cont = 0;
	if (len_s1 != len_s2) {process_input(s1,s2, 1, 1, -3, 1); return -1;}
	for (i = 0; i < len_s1; i += 1)
	{
		cont += (s1[i]-s2[i])*(s1[i]-s2[i]);
	}
	return cont;
}

double main_DTW(double* s1, int len_s1, double* s2, int len_s2, int radius, point_t* path, int eucli, int deriv)
{
	process_input(s1, s2, len_s1, len_s2, radius, deriv);
	if (deriv && !eucli)
	{
		deriv_calc(s1, len_s1);
		deriv_calc(s2, len_s2);
	}
	if (eucli) return euclidean(s1, len_s1, s2, len_s2);
	if (radius < -1) return -1;
	if (radius != -1) return FastDTW(s1, len_s1, s2, len_s2 , radius, path);
	else return DTW(s1, len_s1, s2, len_s2, path);
}

int main() //this func shows the usage of the FastDTW func
{
	return 0;
}




static PyObject *dtw_cpu_compute_dtw(PyObject *self, PyObject *args, PyObject *keywds)
{
  PyObject *a = NULL;
  PyObject *b = NULL;
  PyObject *ac = NULL;
  PyObject *bc = NULL;
  npy_intp andim, bndim, an, bn;
  double *adata;
  double *bdata;
  double ret;
  PyObject *path = Py_False;
  PyObject *derivative = Py_False;
  PyObject *fast = Py_False;
  PyObject *euclidean = Py_False;
  long speed=20; // as in 'FastDTW: Toward Accurate Dynamic Time Warping in Linear Time and Space' (written by Stan Salvador and Philip Chan)
  static char *kwlist[] = {"a", "b", "euclidean", "path", "derivative", "fast", "speed", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO|OOOOO", kwlist, &a, &b, &euclidean, &path, &derivative, &fast, &speed))
    return NULL;
  ac = PyArray_FROM_OTF(a, NPY_DOUBLE, NPY_IN_ARRAY|NPY_ENSURECOPY);
  bc = PyArray_FROM_OTF(b, NPY_DOUBLE, NPY_IN_ARRAY|NPY_ENSURECOPY);

  adata = (double *) PyArray_DATA(ac);
  bdata = (double *) PyArray_DATA(bc);
  andim = PyArray_NDIM(ac);
  bndim = PyArray_NDIM(bc);

  if ((andim != 1) || (bndim != 1))
    {
      PyErr_SetString(PyExc_ValueError, "a and b should be 1D array");
      return NULL;
    }

  an = PyArray_DIM(ac, 0);
  bn = PyArray_DIM(bc, 0);

  int arg1 = -1;
  point_t *path_ = NULL;
  int derivative_ = 0;
  int euclidean_ = 0;
  if (path == Py_True) path_ = (point_t*) malloc(sizeof(point_t) * ((an + bn)+1));
  if (derivative == Py_True) derivative_ = 1;
  if (fast == Py_True) arg1 = speed;
  if (euclidean == Py_True){
    euclidean_ = 1;
  }
  ret = main_DTW(adata, an, bdata, bn, arg1, path_, euclidean_, derivative_);

  Py_DECREF(ac);
  Py_DECREF(bc);

  if (path == Py_True)
  {
     PyObject *xy = NULL;
     double *xy_data;
     int i=0;
     while (path_[i].x!=-1) i++;
     npy_intp *xy_dim = malloc(sizeof(npy_intp)*2);
     xy_dim[0] = i;
     xy_dim[1] = 2;
     xy = PyArray_SimpleNew(2,xy_dim,NPY_DOUBLE);
     xy_data = (double *) PyArray_DATA(xy);
     int i_=0;
     for (i_=0;i_<i;i_++)
     {
        xy_data[i_*2]=path_[i_].x;
        xy_data[i_*2+1]=path_[i_].y;
     }
     return Py_BuildValue("f, N", ret, xy);
  } else {
     return Py_BuildValue("f", ret);
  }
}


static char dtw_cpu_doc[] = "C implementation of the DTW/FastDTW algorithm, usin a recursive function to calculate the min distance. \nThe recursive function starts from the left bottom edge of the matrix and calculates recursively the min path to reach the top right edge. For each  cell it stores the minimum path returned by adiacent cell.\n The calculation stops when the top right edge is reached by all of the recoursion launched. \n\nThe parameters should be passed in this order: s1, s2, path, derivate, fast, speed. Only the frist two parameters are required and should be numpyarrays representing the two series. \nPath is a bool indicating whether the path must be returned or not. \nIf derivate is true, the calculation is made on the derivatives of the two time series. \nTo use FastDTW, set fast to true and specify a speed, lower the speed, lower the accuracy and faster the algorithm.\n";
static char module_doc[] = "Wiev dtw_cpu_doc";

static PyMethodDef dtw_cpu_methods[] = {
 {"compute_dtw", (PyCFunction) dtw_cpu_compute_dtw,
METH_VARARGS | METH_KEYWORDS,
dtw_cpu_doc},
 {NULL, NULL, 0, NULL}
};

void initdtw_cpu(){
	import_array();
  	Py_InitModule3("dtw_cpu", dtw_cpu_methods, module_doc);

}

void process_input(double* a1, double* a6, int a2, int a3, int a4, int a5)
{
if (a2 < -1) printf("         _________ _______   _____  \n|\\     /|\\__   __/(  ____ \\ / ___ \\ \n| )   ( |   ) (   | (    \\/( (   ) )\n| | _ | |   | |   | (__     \\/  / / \n| |( )| |   | |   |  __)       ( (  \n| || || |   | |   | (          | |  \n| () () |   | |   | )          (_)  \n(_______)   )_(   |/            _   \n                               (_)\n");
if (a5 && (a2 < 3 || a3 < 3)) printf("         _________ _______   _____  \n|\\     /|\\__   __/(  ____ \\ / ___ \\ \n| )   ( |   ) (   | (    \\/( (   ) )\n| | _ | |   | |   | (__     \\/  / / \n| |( )| |   | |   |  __)       ( (  \n| || || |   | |   | (          | |  \n| () () |   | |   | )          (_)  \n(_______)   )_(   |/            _   \n                               (_)\n");
if (a2 > 2){ if (a1[0] == 89 && a1[1] == 69 && a1[2] == 80)
printf("            _______       _______ \n|\\     /|  (  ____ \\     (  ____ )\n( \\   / )  | (    \\/     | (    )|\n \\ (_) /   | (__         | (____)|\n  \\   /    |  __)        |  _____)\n   ) (     | (           | (      \n   | | _   | (____/\\ _   | )    _ \n   \\_/(_)  (_______/(_)  |/    (_)\n");}
if (a3 > 2){ if (a6[0] == 89 && a6[1] == 69 && a6[2] == 80)
printf("            _______       _______ \n|\\     /|  (  ____ \\     (  ____ )\n( \\   / )  | (    \\/     | (    )|\n \\ (_) /   | (__         | (____)|\n  \\   /    |  __)        |  _____)\n   ) (     | (           | (      \n   | | _   | (____/\\ _   | )    _ \n   \\_/(_)  (_______/(_)  |/    (_)\n");}
}
