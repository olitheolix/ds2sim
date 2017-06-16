#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    'Augmentor',
    'matplotlib',
    'pillow',
    'requests',
    'tornado',
]

test_requirements = [
    'pytest'
]

setup(
    name='ds2server',
    version='0.3.1',
    description="Fun with Machine Learning and Space Ships",
    long_description=readme + '\n\n',
    author="Oliver Nagy",
    author_email='olitheolix@gmail.com',
    url='https://github.com/olitheolix/ds2server',
    packages=['ds2server'],
    include_package_data=True,
    zip_safe=False,
    license="Apache Software License 2.0",
    keywords='ds2server',
    test_suite='tests',
    scripts=['scripts/ds2generate', 'scripts/ds2server', 'scripts/ds2drawboxes'],
    install_requires=requirements,
    tests_require=test_requirements,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
    ],
)
