from Cython.Build import cythonize
from setuptools import setup
from setuptools.extension import Extension

try:
    import cython  # noqa
    import numpy as np

    include_dirs = [np.get_include(), "correlations"]
except ImportError:
    raise ImportError(
        """Could not import cython or numpy. Building this package from source requires
cython and numpy to be installed. Please install these packages using
the appropriate package manager for your python environment."""
    )

cython_extensions = [
    Extension(
        "py_constr_field.correlations.utils",
        ["py_constr_field/correlations/utils.pyx"],
        include_dirs=include_dirs,
    )
]

setup(ext_modules=cythonize(cython_extensions))
