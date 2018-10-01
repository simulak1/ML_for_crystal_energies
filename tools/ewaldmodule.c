#include <Python.h>

float doublefloat(float M){
  return 2*M;
    }

static PyObject *
DoubleFloatFunc(PyObject *self, PyObject *args)
{
float M;
if (!PyArg_ParseTuple(args, "f", &M))
  return NULL;

 return Py_BuildValue("f",doublefloat(M));
}

static PyMethodDef DoubleMethods[] = {
{"doublef",DoubleFloatFunc, METH_VARARGS, "Execute a shell command"},
{NULL,NULL,0, NULL}
  };

static struct PyModuleDef doublemodule={
PyModuleDef_HEAD_INIT,
  "double",
  NULL,
  -1,
  DoubleMethods
};

PyMODINIT_FUNC
PyInit_doublef(void){
return PyModule_Create(&doublemodule);
}

int main(int argc, char *argv[]){
  wchar_t *program = Py_DecodeLocale(argv[0],NULL);
  if (program==NULL){
    fprintf(stderr,"Fatal error: cannot decode argv[0]\n");
    exit(1);
  }
  PyImport_AppendInittab("doublef",PyInit_doublef);
  
  Py_SetProgramName(program);
  
  Py_Initialize();
  
  PyImport_ImportModule("doublef");
  
  PyMem_RawFree(program);
  return 0;
}
