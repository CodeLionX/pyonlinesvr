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
from typing import Any, ContextManager, Optional

import numpy as np
import scipy as sp
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_array, column_or_1d, check_is_fitted

from pyonlinesvr.lib.compat import (
    double_matrix_to_np,
    double_vector_to_np,
    int_vector_to_np,
    kernel_map,
    kernels,
    np_to_double_matrix,
    np_to_double_vector,
    np_to_int_vector,
)
from pyonlinesvr.lib.onlinesvr import OnlineSVR as LibOnlineSVR


class wrap_output(ContextManager):
    def __init__(self, verbose: int = 0, context: str = "") -> None:
        self.verbose = bool(verbose)
        self.context = context

    def __enter__(self) -> "wrap_output":
        if self.verbose:
            print("\n")
            if self.context:
                print(f"[libonlinesvr] {self.context}")
            print("[libonlinesvr] begin ===========================")
        return self

    def __exit__(self, *args: Any) -> None:
        if self.verbose:
            print("[libonlinesvr] end =============================")


class OnlineSVR(BaseEstimator, RegressorMixin):
    """Epsilon-Support Vector Regression with online learning capability.

    The free parameters in the model are C and epsilon. Support Vector Machine
    algorithms are not scale invariant, so it is highly recommended to scale your data.

    The implementation wraps a C++ code [1]_ implementing an online support vector machine for regression [2]_.

    Parameters
    ----------
    C : float, optional (default=30.0)
        Penalty parameter C of the error term.

    epsilon : float, optional (default=0.1)
         Epsilon in the epsilon-SVR model. It specifies the epsilon-tube within which no
         penalty is associated in the training loss function with points predicted
         within a distance epsilon from the actual
         value.

    kernel : string, optional (default='rbf')
         Specifies the kernel type to be used in the algorithm. It must be one of
         'linear', 'poly', 'rbf', 'rbf-gaussian', 'rbf-exp', or 'sigmoid'.

    degree : int, optional (default=3)
        Degree of the polynomial kernel function ('poly'). Ignored by all other kernels.

    gamma : float, optional (default=None)
        Kernel coefficient for 'poly', 'sigmoid', and 'rbf'-kernels. If gamma is None
        then 1/n_features will be used instead.

    coef0 : float, optional (default=0.0)
        Independent term in kernel function. It is only significant in 'poly' and
        'sigmoid'.

    tol : float, optional (default=1e-3)
        Tolerance for stopping criterion.

    stabilized : bool, optional (default=True)

    save_kernel_matrix : bool, optional (default=True)

    verbose : int, optional default: 0
        Controls verbose output. Higher values mean more detailled output [0; 3].

    Attributes
    ----------
    support_ : array-like, shape = [nSV]
        Indices of support vectors.

    support_vectors_ : array-like, shape = [nSV, n_features]
        Support vectors.

    intercept_ : array, shape = [1]
        Constants in decision function.

    Example
    -------
    >>> import numpy as np
    >>> from pyonlinesvr import OnlineSVR
    >>> X = np.array([[0, 0], [2, 2]])
    >>> y = np.array([0.5, 2.5])
    >>> regr = OnlineSVR()
    >>> regr.fit(X[:1], y[:1])
    OnlineSVR()
    >>> regr.predict([[1, 1]])
    array([ 0.4])
    >>> regr.partial_fit(X[1:], y[1:])
    OnlineSVR()
    >>> regr.predict([[1, 1]])
    array([ 1.5])

    See also
    --------
    sklearn.svm.SVR
        Support Vector Machine for regression implemented using libsvm.

    References
    ----------
    .. [1] `libonlinesvr: Online Support Vector Machines for Regression <https://github.com/fp2556/onlinesvr>`_
    .. [2] `Parrelly, Francesco (2007). "Online Support Vector Machines for Regression." Master thesis. University of Genoa, Italy. <http://onlinesvr.altervista.org/Research/Online%20Support%20Vector%20Regression%20(Parrella%20F.)%20[2007].pdf>`_
    """

    def __init__(
        self,
        C: float = 30.0,
        epsilon: float = 0.1,
        kernel: str = "rbf",
        degree: int = 3,
        gamma: Optional[float] = None,
        coef0: float = 0.0,
        tol: float = 1e-3,
        stabilized: bool = True,
        save_kernel_matrix: bool = True,
        verbose: int = 0,
    ) -> None:
        super().__init__()
        if gamma == 0:
            raise ValueError(
                "A gamma value of 0.0 is invalid. Use 'None' to"
                " set gamma to `1/n_features`."
            )

        if kernel not in kernels:
            raise ValueError(
                f"Wrong Kernel '{kernel}'. Use one of " f"{','.join(kernels)}"
            )

        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.stabilized = stabilized
        self.save_kernel_matrix = save_kernel_matrix
        self.verbose = verbose
        # for type documentation purposes:
        # self._libosvr_: LibOnlineSVR = None
        # self.n_features_in_: Optional[int] = None

    def fit(self, X: Any, y: Any) -> "OnlineSVR":
        """Fit a new SVR model according to the given training data. Use
        ``partial_fit()`` to continue training in an incremental fasion
        (online training).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and n_features is
            the number of features.

        y : array-like, shape (n_samples,)
            Target values (real numbers)

        Returns
        -------
        self : object
            Returns self.

        Notes
        ------
        If X and y are not C-ordered and contiguous arrays of np.float64 and X is not a
        scipy.sparse.csr_matrix, X and/or y may be copied.
        """
        if hasattr(self, "_libosvr_"):
            del self._libosvr_
        return self.partial_fit(X, y)

    def partial_fit(self, X: Any, y: Any) -> "OnlineSVR":
        """Continues/Starts fitting the SVR model according to the given training data
        in an incremental fasion. If this model was already trained on other data, the
        fit is adapted to the new data (online training).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and n_features is
            the number of features.

        y : array-like, shape (n_samples,)
            Target values (real numbers)

        Returns
        -------
        self : object
            Returns self.

        Notes
        ------
        If X and y are not C-ordered and contiguous arrays of np.float64, X and/or y
        may be copied.
        """

        sparse = sp.sparse.isspmatrix(X)
        if sparse:
            raise ValueError("Sparse inputs are not supported.")

        X, y = check_X_y(
            X, y, dtype=np.float64, order="C", accept_sparse=False, y_numeric=True
        )
        y = column_or_1d(y, warn=True).astype(np.float64)

        if not hasattr(self, "_libosvr_"):
            self._init_lib_online_svr(X.shape[1])
        else:
            self._check_X_shape(X)

        # convert to internal data representation
        X = np_to_double_matrix(X)
        y = np_to_double_vector(y)

        with wrap_output(self.verbose, "Fitting"):
            self._libosvr_.Train(X, y)
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Perform regression on samples in X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Samples to perform regression on.

        Returns
        -------
        y_pred : array, shape (n_samples,)

        Notes
        ------
        If X and y are not C-ordered and contiguous arrays of np.float64, X and/or y
        may be copied.
        """
        check_is_fitted(self, ["_libosvr_", "n_features_in_"])
        X = check_array(
            X,
            dtype=np.float64,
            order="C",
            accept_sparse=False,
            accept_large_sparse=False,
        )
        self._check_X_shape(X)

        # convert to internal data representation
        X = np_to_double_matrix(X)

        with wrap_output(self.verbose, "Predicting"):
            predictions = self._libosvr_.Predict(X)

        # convert predictions back
        predictions = double_vector_to_np(predictions)
        return predictions

    def describe(self) -> None:
        """Prints information about this regressor to stdout."""
        if self._libosvr_:
            self._libosvr_.ShowInfo()
            if self.verbose > 1:
                self._libosvr_.ShowDetails()
        else:
            print(f"Uninitialized OnlineSVR with paramters: {self.get_params()}")

    def forget(self, X: Any) -> None:
        """Remove previously learned samples from the fit.

        You can either pass an array of training samples with shape
        (n_samples, n_features) or an array of indices with shape (n_samples,) to this
        function. The array of indices must be of integer type.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Samples to remove from learning. If shape is (n_samples) and X contains only
            integers, X is assumed to contain indices to training samples.

        Notes
        ------
        If X and y are not C-ordered and contiguous arrays of np.float64, X and/or y
        may be copied.
        """
        check_is_fitted(self, ["_libosvr_", "n_features_in_"])
        X = np.asarray(X)
        if len(X.shape) == 1 and np.issubdtype(X.dtype, np.integer):
            return self._forget_indices(X)
        else:
            return self._forget_values(X)

    @property
    def support_(self) -> np.ndarray:
        """Indices of support vectors"""
        check_is_fitted(self, ["_libosvr_", "n_features_in_"])
        support = self._libosvr_.GetSupportSetIndexes()
        return int_vector_to_np(support)

    @property
    def support_vectors_(self) -> np.ndarray:
        """Support vectors"""
        check_is_fitted(self, ["_libosvr_", "n_features_in_"])
        support_vecs = self._libosvr_.GetSupportVectors()
        return double_matrix_to_np(support_vecs)

    @property
    def intercept_(self) -> float:
        """Constant in decision function."""
        check_is_fitted(self, ["_libosvr_", "n_features_in_"])
        return self._libosvr_.GetBias()

    def _init_lib_online_svr(self, n_features: int) -> None:
        self._libosvr_ = LibOnlineSVR()
        self._libosvr_.SetC(self.C)
        self._libosvr_.SetEpsilon(self.epsilon)
        self._libosvr_.SetKernelType(kernel_map[self.kernel])
        if self.gamma is None:
            gamma = 1.0 / n_features
        else:
            gamma = self.gamma
        self._libosvr_.SetKernelParam(gamma)
        self._libosvr_.SetKernelParam2(self.coef0)
        self._libosvr_.SetKernelParam3(self.degree)
        # TODO: infer from tol param
        self._libosvr_.SetAutoErrorTollerance(False)
        self._libosvr_.SetErrorTollerance(self.tol)
        self._libosvr_.SetStabilizedLearning(self.stabilized)
        self._libosvr_.SetSaveKernelMatrix(self.save_kernel_matrix)
        self._libosvr_.SetVerbosity(self.verbose)

        self.n_features_in_ = n_features

    def _check_X_shape(self, X: np.ndarray) -> None:
        if X.shape[1] != self.n_features_in_:  # type: ignore
            raise ValueError(
                f"X.shape[1]={X.shape[1]} should be equal to the "  # type: ignore
                "number of features at first training time "
                f"(={self.n_features_in_})"  # type: ignore
            )

    def _forget_values(self, X: np.ndarray) -> None:
        X = check_array(
            X,
            dtype=np.float64,
            order="C",
            ensure_2d=True,
            accept_sparse=False,
            accept_large_sparse=False,
        )
        self._check_X_shape(X)

        with wrap_output(self.verbose, "Forgetting values"):
            for sample in X:
                # convert to internal data representation
                sample = np_to_double_vector(sample)
                self._libosvr_.Forget(sample)

    def _forget_indices(self, X: Any) -> None:
        X = check_array(
            X,
            dtype=np.int64,
            ensure_2d=False,
            order="C",
            copy=False,
            accept_sparse=False,
            accept_large_sparse=False,
        )

        # convert to internal data representation
        X = np_to_int_vector(X)

        with wrap_output(self.verbose, "Forgetting indices"):
            self._libosvr_.Forget(X)

    def __getstate__(self):
        dd = super().__getstate__()
        if "_libosvr_" in dd:
            import tempfile
            from pathlib import Path

            dd = dd.copy()
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp).resolve()
                path /= "osvr-dump.txt"
                self._libosvr_.SaveOnlineSVR(str(path))
                with path.open("r") as fp:
                    dd["_libosvr_"] = fp.readlines()
        return dd

    def __setstate__(self, state):
        if "_libosvr_" in state:
            import tempfile
            from pathlib import Path

            state = state.copy()
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp).resolve()
                path /= "osvr-load.txt"
                with path.open("w") as fp:
                    fp.writelines(state["_libosvr_"])
                libosvr = LibOnlineSVR()
                libosvr.LoadOnlineSVR(str(path))
                state["_libosvr_"] = libosvr
        return super().__setstate__(state)
