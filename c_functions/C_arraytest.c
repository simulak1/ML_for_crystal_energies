/* A file to test imorting C modules for handling arrays to Python */

#include "Python.h"
#include "numpy/arrayobject.h"
#include "C_arraytest.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/* #### Globals #################################### */

/* ==== Set up the methods table ====================== */
static PyMethodDef _C_arraytestMethods[] = {
  {"make_ewald_matrix", make_ewald_matrix, METH_VARARGS},
  {NULL, NULL}     /* Sentinel - marks the end of this structure */
};

/* ==== Initialize the C_test functions ====================== */
// Module name must be _C_arraytest in compile and linked 
/*void init_C_arraytest()  {
	(void) Py_InitModule("_C_arraytest", _C_arraytestMethods);
	import_array();  // Must be present for NumPy.  Called first after above line.
	}
*/
static struct PyModuleDef _C_arraytest={
  PyModuleDef_HEAD_INIT,
  "_C_arraytest",
  NULL,
  -1,
  _C_arraytestMethods
};


PyMODINIT_FUNC PyInit__C_arraytest(void){
  import_array()
  return PyModule_Create(&_C_arraytest);
}


/* ==== Make a Python Array Obj. from a PyObject, ================                
   generates a double vector w/ contiguous memory which may be a new allocation if
   the original was not a double type or contiguous 
   !! Must DECREF the object returned from this routine unless it is returned to the
   caller of this routines caller using return PyArray_Return(obj) or 
   PyArray_BuildValue with the "N" construct   !!!                                                              
*/
PyArrayObject *pyvector(PyObject *objin)  {
  return (PyArrayObject *) PyArray_ContiguousFromObject(objin,
							NPY_DOUBLE, 1,1);
}

/* ==== Create 1D Carray from PyArray ======================
   Assumes PyArray is contiguous in memory.             */
double *pyvector_to_Carrayptrs(PyArrayObject *arrayin)  {
  int i,n;

  n=arrayin->dimensions[0];
  return (double *) arrayin->data;  /* pointer to arrayin data as double */
}
/* ==== Check that PyArrayObject is a double (Float) type and a vector ==============
   return 1 if an error and raise exception */
int  not_doublevector(PyArrayObject *vec)  {
  if (vec->descr->type_num != NPY_DOUBLE || vec->nd != 1)  {
    PyErr_SetString(PyExc_ValueError,
		    "In not_doublevector: array must be of type Float and 1 dimensional (n).");
    return 1;  }
  return 0;
}

double dot(double x[3], double y[3]){
  double out=0;
  for (int i=0;i<3;i++){
    out=out+x[i]*y[i];
    }
  return out;
}

double *cross(double *x, double *y){
  double *out=malloc(3*sizeof(double));
  out[0]=x[1]*y[2]-x[2]*y[1];
  out[1]=-1.0*(x[0]*y[2]-x[2]*y[0]);
  out[2]=x[0]*y[1]-x[1]*y[0];
  return out;    
}



double *LinearVectorSum(double *x, double *y, double *z, int i, int j, int k){
  double *out=malloc(3*sizeof(double));
  out[0]=i*x[0]+j*y[0]+k*z[0];
  out[1]=i*x[1]+j*y[1]+k*z[1];
  out[2]=i*x[2]+j*y[2]+k*z[2];
  return out;  
}

double short_range(double *xi, double *xj, double Zi, double Zj, double **L, double Lmax, double a){

  double Lx[3],Ly[3],Lz[3],*Lim,*rij,dist,Lx2,Ly2,Lz2,out;
  int Nx,Ny,Nz,i,j,k;

  for (i=0;i<3;i++){
  Lx[i]=L[0][i];
  Ly[i]=L[1][i];
  Lz[i]=L[2][i];
  }
  
  Lx2=sqrt(dot(Lx,Lx));
  Ly2=sqrt(dot(Ly,Ly));
  Lz2=sqrt(dot(Lz,Lz));

  Nx=(Lmax-fmod(Lmax,Lx2))/Lx2+2;
  Ny=(Lmax-fmod(Lmax,Ly2))/Ly2+2;
  Nz=(Lmax-fmod(Lmax,Lz2))/Lz2+2;
  
  out=0;
  
  for (i=-Nx;i<Nx;i++){
    for (j=-Ny;j<Ny;j++){
      for (k=-Nz;k<Nz;k++){
	
	Lim=LinearVectorSum(Lx,Ly,Lz,i,j,k);
	rij=LinearVectorSum(Lim,xi,xj,-1,1,-1);
	dist=sqrt(dot(rij,rij));
	if(dist<Lmax){

	  out=out+erfc(a*dist)/dist;
	}
      }
    }
    }

  out=Zi*Zj*out;
  return out;
}

double long_range(double *xi, double *xj, double Zi, double Zj,double G[3][3],double Gmax, double a, double V){

  int Nx,Ny,Nz,i,j,k;
  double Gx[3],Gy[3],Gz[3],Gx2,Gy2,Gz2,*Gim,Gim2,pi,*rij,dist,out,d;

  pi=acos(-1.0);
  
  for (i=0;i<3;i++){
  Gx[i]=G[0][i];
  Gy[i]=G[1][i];
  Gz[i]=G[2][i];
  }
  Gx2=sqrt(dot(Gx,Gx));
  Gy2=sqrt(dot(Gy,Gy));
  Gz2=sqrt(dot(Gz,Gz));
  
  Nx=(Gmax-fmod(Gmax,Gx2))/Gx2+2;
  Ny=(Gmax-fmod(Gmax,Gy2))/Gy2+2;
  Nz=(Gmax-fmod(Gmax,Gz2))/Gz2+2;
  
  rij=LinearVectorSum(Gx,xi,xj,0,1,-1);
  
  out=0;
  for (i=-Nx;i<Nx;i++){
    for (j=-Ny;j<Ny;j++){
      for (k=-Nz;k<Nz;k++){
	Gim=LinearVectorSum(Gx,Gy,Gz,i,j,k);
	Gim2=dot(Gim,Gim);
	dist=sqrt(Gim2);
	if(dist<Gmax && abs(dist)>0.0){
	  d=1/Gim2;
	  out=out+exp(-1.0*Gim2/(4*pow(a,2)))*cos(dot(Gim,rij));///pow(dist,2);
	}
	free(Gim);
      }
    }
  }
  out=out*Zi*Zj/(pi*V);
  return out;
}

static PyObject *make_ewald_matrix(PyObject *self, PyObject *args){

  PyArrayObject *L,*xyz,*Z, *matout;
  double **cL,**cxyz,*cZ, **cout, *Lx,*Ly,*Lz, G[3][3],*Gx,*Gy,*Gz, Lmax, Gmax, a,pi,V;
  int dims[2],Natoms;
  
  /* Parse tuples separately since args will differ between C fcns */
  if (!PyArg_ParseTuple(args, "iO!O!O!ddd",
			&Natoms,&PyArray_Type, &L,&PyArray_Type, &xyz,&PyArray_Type, &Z,&Lmax,&Gmax,&a))  return NULL;
  if (NULL == L)  return NULL;
  if (NULL == xyz)  return NULL;
  if (NULL == Z)   return NULL;
  
  /* Check that object input is 'double' type and a matrix                                                  
     Not needed if python wrapper function checks before call to this routine */

  if (not_doublematrix(L)) return NULL;
  if (not_doublematrix(xyz)) return NULL;
  
  
  /* Get the dimensions of the input */
  dims[0]=dims[1]=80; /// This is the number of atoms in the largest system!
  /* Make a new double matrix of same dims */
  matout=(PyArrayObject *) PyArray_FromDims(2,dims,NPY_DOUBLE);
  
  /* Change contiguous arrays into C ** arrays (Memory is Allocated!) */
  cL=pymatrix_to_Carrayptrs(L);
  cxyz=pymatrix_to_Carrayptrs(xyz);
  cZ=pyvector_to_Carrayptrs(Z);
  
  cout=pymatrix_to_Carrayptrs(matout);

  /* Calculate the volume of the simulation cell  */
  Lx=cL[0];
  Ly=cL[1];
  Lz=cL[2];
  V=dot(Lx,cross(Ly,Lz));

  /* Calculate the reciprocal lattice vectors */
  Gx=cross(Ly,Lz);
  Gy=cross(Lz,Lx);
  Gz=cross(Lx,Ly);
  pi=acos(-1.0);
    for (int i=0;i<3;i++){
      G[0][i]=2.0*pi*Gx[i]/V;
      G[1][i]=2.0*pi*Gy[i]/V;
      G[2][i]=2.0*pi*Gz[i]/V;
    }

  
  /* Do the calculation. */
///#pragma omp parallel for
  for (int i=0; i<80; i++)  {
    for (int j=i+1; j<80; j++)  {
      cout[i][j]=0.0;
      if(i<Natoms && j<Natoms){
	//cout[i][j]+= short_range(cxyz[i],cxyz[j],cZ[i],cZ[j],cL,Lmax,a);
	cout[i][j]+=long_range(cxyz[i],cxyz[j],cZ[i],cZ[j],G,Gmax,a,V);
	cout[j][i]=cout[i][j];
	 }
    }
  }

  /* Free memory, close file and return */
  free_Carrayptrs(cL);
  free_Carrayptrs(cxyz);
  free_Carrayptrs(cout);
  ///free_Carrayptrs(cZ);
  return PyArray_Return(matout);
  
  
}

PyArrayObject *pymatrix(PyObject *objin)  {
	return (PyArrayObject *) PyArray_ContiguousFromObject(objin,
		NPY_DOUBLE, 2,2);
}
/* ==== Create Carray from PyArray ======================
    Assumes PyArray is contiguous in memory.
    Memory is allocated!                                    */
double **pymatrix_to_Carrayptrs(PyArrayObject *arrayin)  {
	double **c, *a;
	int i,n,m;
	
	n=arrayin->dimensions[0];
	m=arrayin->dimensions[1];
	c=ptrvector(n);
	a=(double *) arrayin->data;  /* pointer to arrayin data as double */
	for ( i=0; i<n; i++)  {
		c[i]=a+i*m;  }
	return c;
}
/* ==== Allocate a double *vector (vec of pointers) ======================
    Memory is Allocated!  See void free_Carray(double ** )                  */
double **ptrvector(long n)  {
	double **v;
	v=(double **)malloc((size_t) (n*sizeof(double)));
	if (!v)   {
		printf("In **ptrvector. Allocation of memory for double array failed.");
		exit(0);  }
	return v;
}
/* ==== Free a double *vector (vec of pointers) ========================== */ 
void free_Carrayptrs(double **v)  {
	free((char*) v);
}
/* ==== Check that PyArrayObject is a double (Float) type and a matrix ==============
    return 1 if an error and raise exception */ 
int  not_doublematrix(PyArrayObject *mat)  {
  if (mat->descr->type_num != NPY_DOUBLE || mat->nd != 2)  {
    PyErr_SetString(PyExc_ValueError,
		    "In not_doublematrix: array must be of type Float and 2 dimensional (n x m).");
    return 1;
  }
  return 0;
}

