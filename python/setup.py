'''
Created on 27 May 2015

@author: jadarve
'''

from distutils.core import setup, Extension
from Cython.Build import cythonize

import numpy as np

# FIXME: I need a way to set this from CMake
incDirs = ['/usr/local/include',
           '/usr/include/eigen3',
           '/usr/local/cuda/include',
           np.get_include()]

libDirs = ['/usr/local/lib']

# TODO: set externally by CMake
libs = ['spherepix']
libs_gpu = ['flowfilter_gpu', 'spherepix', 'spherepix_gpu']
cflags = ['-std=c++11']

cython_directives = {'embedsignature' : True}

def createExtension(name, sources, libs=libs):

    global incDirs
    global libDirs

    ext = Extension(name,
                    sources=sources,
                    include_dirs=incDirs,
                    library_dirs=libDirs,
                    libraries=libs,
                    runtime_library_dirs=libs,
                    language='c++',
                    extra_compile_args=cflags)

    return ext

extensions = list()

#################################################
# DEFAULT PACKAGE
#################################################
modulesTable = [('spherepix.image', ['spherepix/image.pyx']),
                ('spherepix.eigen', ['spherepix/eigen.pyx']),
                ('spherepix.geometry', ['spherepix/geometry.pyx']),
                ('spherepix.pixelation', ['spherepix/pixelation.pyx']),
                ('spherepix.springdynamics', ['spherepix/springdynamics.pyx']),
                ('spherepix.camera', ['spherepix/camera.pyx'])
                ]

for mod in modulesTable:

    extList = cythonize(createExtension(mod[0], mod[1], libs=libs),
        compiler_directives=cython_directives)

    extensions.extend(extList)

#################################################
# GPU PACKAGE
#################################################
GPU_BUILD = True

if  GPU_BUILD:

    GPUmodulesTable = [('spherepix.gpu.pixelation', ['spherepix/gpu/pixelation.pyx']),
                       ('spherepix.gpu.camera', ['spherepix/gpu/camera.pyx']),
                       ('spherepix.gpu.pyramid', ['spherepix/gpu/pyramid.pyx'])
                      ]

    for mod in GPUmodulesTable:

        extList = cythonize(createExtension(mod[0], mod[1], libs=libs_gpu),
            compiler_directives=cython_directives)

        extensions.extend(extList)


#################################################
# PURE PYTHON PACKAGES
#################################################
py_packages = ['spherepix', 'spherepix.gpu', 'spherepix.plot']

# package data include Cython .pxd files
package_data = {'spherepix' : ['*.pxd'],
                'spherepix.gpu': ['*.pxd']}

# call distutils setup
setup(name='spherepix',
    version='0.1',
    author='Juan David Adarve',
    author_email='juanda0718@gmail.com',
    maintainer='Juan David Adarve',
    maintainer_email='juanda0718@gmail',
    url='',
    description='A framework for spherical image processing',
    license='3-clause BSD',
    packages=py_packages,
    ext_modules=extensions,
    package_data=package_data)