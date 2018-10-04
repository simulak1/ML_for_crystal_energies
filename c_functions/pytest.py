import numpy as np
import _C_arraytest

A=np.ones((2,2))

print(A)

print(_C_arraytest.make_ewald_matrix(A))

