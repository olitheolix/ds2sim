# -*- coding: utf-8 -*-

__author__ = """Oliver Nagy"""
__email__ = 'olitheolix@gmail.com'
__version__ = '0.1.0'

import os
import sys

# Path to Cython generated libraries, and Protocol buffers.
fpath = os.path.dirname(os.path.abspath(__file__))
libpath = os.path.join(fpath, '..', 'lib')
sys.path.insert(0, libpath)
del fpath, libpath
