# ---- Link ---------------------------
_C_arraytest.so:  C_arraytest.o  C_arraytest.mak
	gcc -bundle -flat_namespace -undefined suppress -o _C_arraytest.so  C_arraytest.o

# ---- gcc C compile ------------------
C_arraytest.o:  C_arraytest.c C_arraytest.h C_arraytest.mak
	gcc -c C_arraytest.c -I/Library/Frameworks/Python.framework/Versions/2.4/include/python2.4 -I/Library/Frameworks/Python.framework/Versions/2.4/lib/python2.4/site-packages/numpy/core/include/numpy/
