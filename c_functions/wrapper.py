from distutils.core import setup,Extension
import numpy

module1=Extension('_C_arraytest',sources=['C_arraytest.c'],extra_compile_args=['-fopenmp'],extra_link_args=['-lgomp'])

setup(name='double',version='1.0',description='This is a test package',ext_modules=[module1])
