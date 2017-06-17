# -*- coding: utf-8 -*-

import os

__version__ = '0.3.1'


def getResourcePath():
    """ Return absolute path to Horde's default resources.

    The default resources are verbatim copies of those in the original
    Horde3D repository. These are useful for small demos to ensure the
    engine works.
    """
    path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(path, 'resources')
