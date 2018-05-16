from setuptools import setup, Extension
from Cython.Build import cythonize

setup(
    name='phys',
    version='0.0.1',
    description='',
    install_requires=[
        'setuptools>=27',
    ],
    keywords='physics',
    ext_modules=cythonize([
        Extension(
            'phys',
            sources=['phys.pyx']
        )
    ])
)
