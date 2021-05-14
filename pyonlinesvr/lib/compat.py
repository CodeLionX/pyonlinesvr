from typing import Any, Dict, List

from pyonlinesvr.lib.onlinesvr import (
    IntVector,
    IntMatrix,
    OnlineSVR,
    DoubleMatrix,
    DoubleVector,
)
from sklearn.utils.validation import check_array
import numpy as np


kernel_map: Dict[str, Any] = {
    "linear": OnlineSVR.KERNEL_LINEAR,
    "sigmoid": OnlineSVR.KERNEL_MLP,
    "poly": OnlineSVR.KERNEL_POLYNOMIAL,
    "rbf": OnlineSVR.KERNEL_RBF,
    "rbf-exp": OnlineSVR.KERNEL_RBF_EXPONENTIAL,
    "rbf-gaussian": OnlineSVR.KERNEL_RBF_GAUSSIAN,
}

kernels: List[str] = list(kernel_map.keys())


def np_to_double_vector(arr: np.ndarray) -> DoubleVector:
    arr = check_array(
        arr,
        dtype=np.float64,
        ensure_2d=False,
        order="C",
        copy=False,
        accept_sparse=False,
        accept_large_sparse=False,
    )
    if len(arr.shape) > 1:
        raise ValueError(
            "Can only convert a 1-dimensional array into a DoubleVector! Input array "
            f"has shape={arr.shape}"
        )

    vec = DoubleVector()
    for v in arr:
        vec.Add(v)
    return vec


def double_vector_to_np(vec: DoubleVector) -> np.ndarray:
    length = vec.GetLength()
    arr = np.empty(length, dtype=np.float64, order="C")
    for i in range(length):
        arr[i] = vec.GetValue(i)
    return arr


def np_to_double_matrix(arr: np.ndarray) -> DoubleMatrix:
    arr = check_array(
        arr,
        dtype=np.float64,
        ensure_2d=True,
        order="C",
        copy=False,
        accept_sparse=False,
        accept_large_sparse=False,
    )
    m = DoubleMatrix()
    for row in arr:
        row_vec = np_to_double_vector(row)
        m.AddRowCopy(row_vec)
    return m


def double_matrix_to_np(m: DoubleMatrix) -> np.ndarray:
    rows = m.GetLengthRows()
    cols = m.GetLengthCols()

    arr = np.empty((rows, cols), dtype=np.float64, order="C")
    for i in range(rows):
        for j in range(cols):
            arr[i, j] = m.GetValue(i, j)
    return arr


def np_to_int_vector(arr: np.ndarray) -> IntVector:
    arr = check_array(
        arr,
        dtype=np.int64,
        ensure_2d=False,
        order="C",
        copy=False,
        accept_sparse=False,
        accept_large_sparse=False,
    )
    if len(arr.shape) > 1:
        raise ValueError(
            "Can only convert a 1-dimensional array into a IntVector! Input array "
            f"has shape={arr.shape}"
        )

    vec = IntVector()
    for v in arr:
        vec.Add(int(v))
    return vec


def int_vector_to_np(vec: IntVector) -> np.ndarray:
    length = vec.GetLength()
    arr = np.empty(length, dtype=np.int64, order="C")
    for i in range(length):
        arr[i] = vec.GetValue(i)
    return arr


def np_to_int_matrix(arr: np.ndarray) -> IntMatrix:
    arr = check_array(
        arr,
        dtype=np.int64,
        ensure_2d=True,
        order="C",
        copy=False,
        accept_sparse=False,
        accept_large_sparse=False,
    )
    m = IntMatrix()
    for row in arr:
        row_vec = np_to_int_vector(row)
        m.AddRowCopy(row_vec)
    return m


def int_matrix_to_np(m: IntMatrix) -> np.ndarray:
    rows = m.GetLengthRows()
    cols = m.GetLengthCols()

    arr = np.empty((rows, cols), dtype=np.int64, order="C")
    for i in range(rows):
        for j in range(cols):
            arr[i, j] = m.GetValue(i, j)
    return arr
