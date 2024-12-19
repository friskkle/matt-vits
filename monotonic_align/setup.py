from distutils.core import setup
from Cython.Build import cythonize
import numpy
import os

# Ensure the output directory exists
os.makedirs("monotonic_align", exist_ok=True)

setup(
  name = 'monotonic_align',
  ext_modules = cythonize("core.pyx"),
  include_dirs=[numpy.get_include()]
)
