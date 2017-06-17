#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize


def isHordeInstalled():
    fnames = [
        'lib/libHorde3D.so', 'lib/libHorde3DUtils.so',
        'include/horde3d',
        'include/horde3d/Horde3D.h',
        'include/horde3d/Horde3DUtils.h',
    ]

    ok = True
    for fname in fnames:
        if not os.path.exists(os.path.join(sys.prefix, fname)):
            print(f'Error: cannot find {fname}')
            ok = False

    if not ok:
        p = sys.prefix
        print('\nHorde3D does not appear to be installed.')
        print('Install it as follows:')
        print(' >> git clone https://github.com/olitheolix/Horde3D')
        print(' >> mkdir -p Horde3D/build')
        print(' >> cd Horde3D/build')
        print(' >> git checkout ds2')
        print(f' >> cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX={p}')
        print(' >> make install')
    return ok


def main():
    if not isHordeInstalled():
        sys.exit(1)

    pyhorde = Extension(
        name='pyhorde',
        sources=['cython/pyhorde.pyx', 'cython/glutils.cpp'],
        include_dirs=[os.path.join(sys.prefix, 'include')],
        library_dirs=[os.path.join(sys.prefix, 'lib')],
        libraries=['Horde3D', 'Horde3DUtils', 'EGL'],
        language='c++',
        extra_compile_args=['-std=c++14'],
        extra_objects=[],
    )

    setup(
        name='ds2sim',
        version='0.3.1',
        description="Fun with Machine Learning and Space Ships",
        long_description=open('README.md').read() + '\n\n',
        author="Oliver Nagy",
        author_email='olitheolix@gmail.com',
        url='https://github.com/olitheolix/ds2sim',
        packages=['ds2sim'],
        include_package_data=True,
        zip_safe=False,
        license="Apache Software License 2.0",
        keywords='ds2sim',
        test_suite='tests',
        scripts=['scripts/ds2server'],
        install_requires=['pillow', 'requests', 'tornado'],
        tests_require=['pytest'],
        classifiers=[
            'Development Status :: 2 - Pre-Alpha',
            'Intended Audience :: Developers',
            'License :: OSI Approved :: Apache Software License',
            'Natural Language :: English',
            'Programming Language :: Python :: 3.6',
        ],
        ext_modules=cythonize([pyhorde]),
    )


if __name__ == '__main__':
    main()
