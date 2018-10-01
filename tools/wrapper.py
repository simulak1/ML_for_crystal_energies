from distutils.core import setup,Extension

module1=Extension('doublef',sources=['ewaldmodule.c'])

setup(name='doublef',version='1.0',description='This is a test package',ext_modules=[module1])
