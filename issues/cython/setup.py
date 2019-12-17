"""
python setup.py build_ext --inplace
"""
from distutils.core import setup
from Cython.Build import cythonize

setup(
        ext_modules = cythonize([
            "cy_t3.pyx",
            "cy_def_t3.pyx"
            ], language_level=3)
)
