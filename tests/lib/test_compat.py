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

import pytest
import numpy as np
from pyonlinesvr.lib.compat import (
    kernel_map,
    kernels,
    np_to_double_vector,
    double_vector_to_np,
    np_to_double_matrix,
    double_matrix_to_np,
    np_to_int_vector,
    int_vector_to_np,
    np_to_int_matrix,
    int_matrix_to_np,
)

y = np.sin(np.arange(0, 8, 0.1))
X = y.reshape(-1, 2)

yi = np.arange(0, 8)
Xi = yi.reshape(-1, 2)


def test_kernels():
    ks = [k for k in kernel_map]
    np.testing.assert_array_equal(ks, kernels)


def test_double_vector_conversion():
    v = np_to_double_vector(y)
    new_y = double_vector_to_np(v)
    np.testing.assert_array_equal(y, new_y)


def test_double_vector_wrong_shape():
    with pytest.raises(ValueError, match="Can only convert a 1-dimensional array"):
        np_to_double_vector(X)


def test_int_vector_conversion():
    v = np_to_int_vector(yi)
    new_yi = int_vector_to_np(v)
    np.testing.assert_array_equal(yi, new_yi)


def test_int_vector_wrong_shape():
    with pytest.raises(ValueError, match="Can only convert a 1-dimensional array"):
        np_to_int_vector(Xi)


def test_double_matrix_conversion():
    m = np_to_double_matrix(X)
    new_X = double_matrix_to_np(m)
    np.testing.assert_array_equal(X, new_X)


def test_int_matrix_conversion():
    m = np_to_int_matrix(Xi)
    new_Xi = int_matrix_to_np(m)
    np.testing.assert_array_equal(Xi, new_Xi)
