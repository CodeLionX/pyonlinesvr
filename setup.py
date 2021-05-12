#!/usr/bin/env python

#                         PyOnlineSVR
#               Copyright 2021 - Sebastian Schmidl
#
# This file is part of PyOnlineSVR.
#
# PyOnlineSVR is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyOnlineSVR is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyOnlineSVR. If not, see
# <https://www.gnu.org/licenses/gpl-3.0.html>.


# This future is needed to print Python2 EOL message
from __future__ import print_function
import sys
if sys.version_info < (3,):
    print("Python 2 has reached end-of-life and is not supported by pyonlinesvr.")
    sys.exit(-1)

import platform
python_min_version = (3, 6, 2)
python_min_version_str = '.'.join(map(str, python_min_version))
if sys.version_info < python_min_version:
    print(f"You are using Python {platform.python_version()}. Python >={python_min_version_str} is required.")
    sys.exit(-1)


from setuptools import setup, Extension, find_packages
import os

cwd = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(cwd, "pyonlinesvr", "lib")


onlinesvr_module = Extension('_onlinesvr',
    sources=list(map(lambda x: os.path.join(lib_path, x), [
        "CrossValidation.cpp", "File.cpp", "Forget.cpp", "Kernel.cpp",
        "OnlineSVR.cpp", "Show.cpp", "Stabilize.cpp", "Train.cpp",
        "Variations.cpp", "OnlineSVR_wrap.cxx"
    ])),
)

setup(name = 'PyOnlineSVR',
      version = '0.0.1',
      author      = "Sebastian Schmidl",
      description = """Experiment using swig to interface C++.""",
      ext_modules = [onlinesvr_module],
      py_modules = ["pyonlinesvr"],
)
