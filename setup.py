from distutils.core import setup, Extension
from distutils.sysconfig import *
from distutils.util import *
import os
import numpy

data_files = []

# Include gsl dlls for the win32 distribution
#if get_platform() == "win32":
#    dlls = ["mlpy\gslwin\libgsl-0.dll", "mlpy\gslwin\libgslcblas-0.dll"]
#    data_files += [("Lib\site-packages\mlpy", dlls)]

## Python include
py_include = get_python_inc()

## Numpy header files
numpy_lib = os.path.split(numpy.__file__)[0]
numpy_include = os.path.join(numpy_lib, 'core/include')


##### Includes ######################################################################

base_include  = [py_include, numpy_include]

## dtw include
dtw_include = base_include

#####################################################################################


##### Sources #######################################################################

## dtw sources
dtw_sources = ['lib/dtw/dtw_py.c']
dtw_cpu_sources = ['lib/dtw/dtw_cpu_py.c']

#####################################################################################


## Extra compile args
extra_compile_args = ['-Wno-strict-prototypes']

# Setup
setup(name = 'yeps-clustering-lib',
      version = '0.1.0',
      requires = ['numpy (>= 1.1.0)'],
      description = 'clustering libs for yeps tools',
      author = 'Irish, Flash, fox, Pollo & WebValley Devs',
      author_email = '',
      url = '',
      download_url = '',
      license='GPLv3',
      classifiers=['Development Status :: 5 - Production/Stable',
                   'Intended Audience :: Science/Research',
                   'Intended Audience :: Developers',
                   'License :: OSI Approved :: GNU General Public License (GPL)',
                   'Natural Language :: English',
                   'Operating System :: POSIX :: Linux',
                   'Operating System :: POSIX :: BSD',
                   'Operating System :: Unix',
                   'Operating System :: MacOS :: MacOS X',
                   'Operating System :: Microsoft :: Windows',
                   'Programming Language :: C',
                   'Programming Language :: Python',
                   ],
      package_dir = {'': 'lib'},
      packages=['iodata', 'kmedoid'],
      ext_modules=[Extension('dtw', dtw_sources,
                             include_dirs=dtw_include,
                             extra_compile_args=extra_compile_args),
                   ],
      )
"""
setup(name = 'YEP',
      version = '0.1.0',
      requires = ['numpy (>= 1.1.0)'],
      description = 'YEP Sonification',
      author = 'WebValley Developers',
      author_email = '',
      url = '',
      download_url = '',
      license='GPLv3',
      classifiers=['Development Status :: 5 - Production/Stable',
                   'Intended Audience :: Science/Research',
                   'Intended Audience :: Developers',
                   'License :: OSI Approved :: GNU General Public License (GPL)',
                   'Natural Language :: English',
                   'Operating System :: POSIX :: Linux',
                   'Operating System :: POSIX :: BSD',
                   'Operating System :: Unix',
                   'Operating System :: MacOS :: MacOS X',
                   'Operating System :: Microsoft :: Windows',
                   'Programming Language :: C',
                   'Programming Language :: Python',
                   ],
      package_dir = {'': 'lib'},
      packages=['iodata', 'kmedoid', 'pymplib', 'soni', 'maps', 'action_queue'],
      ext_modules=[Extension('dtw_cpu', dtw_cpu_sources,
                             include_dirs=dtw_include,
                             extra_compile_args=extra_compile_args),
                   ],
      scripts=['scripts/yeps', 'scripts/kmean'],
      data_files = data_files
      )
"""
