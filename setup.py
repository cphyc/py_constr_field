# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

try:
    import numpy as np
    import cython
    include_dirs = [np.get_include(), 'py_constr_field']
except ImportError:
    raise ImportError(
"""Could not import cython or numpy. Building this package from source requires
cython and numpy to be installed. Please install these packages using
the appropriate package manager for your python environment.""")

cython_extensions = [
    # Extension("py_constr_field.utils",
    #           ["py_constr_field/*.pyx"],
    #           include_dirs=include_dirs)
]


# with open('Readme.md') as f:
#     readme = f.read()

# with open('LICENSE') as f:
#     license = f.read()

setup(
    name='py_constr_field',
    version='0.0.1',
    description='Compute correlations functions for Gaussian Random Fields.',
    classifiers=[
        'Development status :: 1 - Alpha',
        'License :: CC-By-SA2.0',
        'Programming Language :: Python',
        'Topic :: Gaussian Random Field'
    ],
    author='Corentin Cadiou',
    author_email='contact@cphyc.me',
    packages=['py_constr_field'],
    package_dir={'py_constr_field': 'py_constr_field'},
    package_data={'py_constr_field': [
        'py_constr_field/*.pyx'
    ]},
    install_requires=[
                      'numpy',
                      'attrs',
                      'opt_einsum'
    ],
    extras_require={
        'dev': ['nose', 'nose-timer']
    },
    include_package_data=True,
    ext_modules=cythonize(cython_extensions)
)
