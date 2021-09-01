from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np

# import line_profiler
# profile = line_profiler.LineProfiler()

####################
# Extra stuff to profile code using kernprof -l -v file.py after it is compiled
# Taken from https://stackoverflow.com/questions/28301931/how-to-profile-cython-functions-line-by-line

# # from Cython.Compiler.Options import get_directive_defaults
# # directive_defaults = get_directive_defaults()
# import Cython
# directive_defaults = Cython.Compiler.Options.get_directive_defaults()
#
# directive_defaults['linetrace'] = True
# directive_defaults['binding'] = True

###########################################


extensions = [
    Extension("c_kepler._kepler",
             ['c_kepler/_kepler.pyx'],
             include_dirs=[np.get_include()], define_macros=[('CYTHON_TRACE', '1')]),
    Extension("trends.log_likelihood",
             ['trends/log_likelihood.pyx'],
             include_dirs=[np.get_include()], define_macros=[('CYTHON_TRACE', '1')]),
    Extension("trends.helper_functions_general",
             ['trends/helper_functions_general.pyx'],
             include_dirs=[np.get_include()], define_macros=[('CYTHON_TRACE', '1')]),
    Extension("trends.helper_functions_rv",
             ['trends/helper_functions_rv.pyx'],
             include_dirs=[np.get_include()], define_macros=[('CYTHON_TRACE', '1')]),
    Extension("trends.helper_functions_astro",
             ['trends/helper_functions_astro.pyx'],
             include_dirs=[np.get_include()], define_macros=[('CYTHON_TRACE', '1')])
    
]

##############################

setup(
    ext_modules = cythonize(extensions, language_level='3', annotate=True)
)