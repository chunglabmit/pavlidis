from setuptools import setup, Extension
import os

VERSION = "0.1.0"

README = open('README.md').read()


def make_ext(modname, pyxfilename):
    import numpy as np
    return Extension(name=modname,
                     sources=[pyxfilename],
                     language="c++",
                     extra_compile_args=["-std=c++11"],
                     include_dirs=[np.get_include()])


setup(
    name="pavlidis",
    version=VERSION,
    packages=["pavlidis"],
    url="https://github.com/chunglabmit/pavlidis",
    description="Implementation of the pavlidis algorithm for finding contours",
    long_description=README,
    install_requires=[
        "Cython",
        "numpy"
    ],
    ext_modules=[make_ext("pavlidis._pavlidis",
                          os.path.join("pavlidis","_pavlidis.pyx"))]
)