#                         PyOnlineSVR
#               Copyright 2023 - Sebastian Schmidl
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

from pathlib import Path

import numpy as np
import pytest
from joblib import load, dump
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import check_estimator

from pyonlinesvr import OnlineSVR

a = np.sin(np.arange(0, 8, 0.1))
X = a.reshape(-1, 1)[:-1]
y = np.roll(a, -1)[:-1]
del a


def test_estimator():
    check_estimator(OnlineSVR())


def test_init():
    rgr = OnlineSVR()
    assert rgr.C == 30.0
    assert rgr.epsilon == 0.1
    assert rgr.kernel == "rbf"
    assert rgr.degree == 3
    assert rgr.gamma is None
    assert rgr.coef0 == 0.0
    assert rgr.tol == 1e-3
    assert rgr.stabilized is True
    assert rgr.save_kernel_matrix is True
    assert rgr.verbose == 0


def test_gamma():
    rgr = OnlineSVR(gamma=None)
    rgr._init_lib_online_svr(4)
    assert rgr.gamma is None
    assert rgr._libosvr_.GetKernelParam() == 0.25

    rgr = OnlineSVR(gamma=1.2)
    rgr._init_lib_online_svr(4)
    assert rgr.gamma == 1.2
    assert rgr._libosvr_.GetKernelParam() == 1.2


def test_init_wrong_gamma():
    with pytest.raises(ValueError, match=r"[A|a] gamma value of 0(\.0)* is invalid"):
        OnlineSVR(gamma=0)


def test_init_wrong_kernel():
    with pytest.raises(ValueError, match=r"[W|w]rong [K|k]ernel"):
        OnlineSVR(kernel="non-existent")


def test_fit():
    rgr = OnlineSVR(C=0.1, verbose=0)
    rgr.fit(X, y)

    assert hasattr(rgr, "_libosvr_")
    assert hasattr(rgr, "n_features_in_")
    assert rgr.n_features_in_ == X.shape[1]
    assert rgr._libosvr_.GetC() == 0.1
    assert rgr._libosvr_.GetVerbosity() == 0
    assert rgr._libosvr_.GetSamplesTrainedNumber() == len(X)


def test_fit_wrong_X_shape():
    rgr = OnlineSVR()
    rgr.partial_fit(X[:5], y[:5])
    with pytest.raises(
        ValueError,
        match=r"X\.shape.* should be equal to the number of features at first training time",
    ):
        rgr.partial_fit(X[5:15].reshape(-1, 2), y[5:10])


def test_partial_fit_begin():
    rgr = OnlineSVR(C=0.1, verbose=0)
    rgr.partial_fit(X, y)

    assert hasattr(rgr, "_libosvr_")
    assert hasattr(rgr, "n_features_in_")
    assert rgr.n_features_in_ == X.shape[1]
    assert rgr._libosvr_.GetSamplesTrainedNumber() == len(X)


def test_partial_fit_continue():
    rgr = OnlineSVR(C=0.1, verbose=0)
    rgr.fit(X[:5], y[:5])

    assert hasattr(rgr, "_libosvr_")
    assert hasattr(rgr, "n_features_in_")
    assert rgr.n_features_in_ == X.shape[1]
    assert rgr._libosvr_.GetSamplesTrainedNumber() == 5

    rgr.partial_fit(X[5:], y[5:])
    assert rgr.n_features_in_ == X.shape[1]
    assert rgr._libosvr_.GetSamplesTrainedNumber() == len(X)


def test_properties():
    rgr = OnlineSVR(C=0.1, verbose=0)
    with pytest.raises(NotFittedError):
        rgr.support_
    with pytest.raises(NotFittedError):
        rgr.support_vectors_
    with pytest.raises(NotFittedError):
        rgr.intercept_

    rgr.fit(X, y)
    assert hasattr(rgr, "support_")
    assert hasattr(rgr, "support_vectors_")
    assert hasattr(rgr, "intercept_")


def test_predict():
    rgr = OnlineSVR(epsilon=1e-3, verbose=0)
    rgr.fit(X[:-5], y[:-5])
    y_hat = rgr.predict(X[-5:])
    np.testing.assert_array_almost_equal(y[-5:], y_hat, decimal=2)


def test_forget():
    rgr = OnlineSVR(epsilon=1e-3, verbose=0)
    rgr.fit(X, y)
    rgr.forget(X[-5:])
    y_hat = rgr.predict(X[-5:])
    np.testing.assert_array_almost_equal(y[-5:], y_hat, decimal=2)


def test_forget_indices():
    rgr = OnlineSVR(epsilon=1e-3, verbose=0)
    rgr.fit(X, y)
    rgr.forget(np.arange(X.shape[0] - 5, X.shape[0], dtype=np.int_))
    y_hat = rgr.predict(X[-5:])
    np.testing.assert_array_almost_equal(y[-5:], y_hat, decimal=2)


def test_pickle_predict(tmp_path: Path) -> None:
    filename = tmp_path / "svr.pkl"
    rgr = OnlineSVR(epsilon=1e-3, verbose=0)
    rgr.fit(X[:-5], y[:-5])
    dump(rgr, filename)
    del rgr

    loaded_rgr = load(filename)
    y_hat = loaded_rgr.predict(X[-5:])
    assert loaded_rgr._libosvr_.GetSamplesTrainedNumber() == len(X[:-5])
    np.testing.assert_array_almost_equal(y[-5:], y_hat, decimal=2)


def test_pickle_refit(tmp_path: Path) -> None:
    filename = tmp_path / "svr2.pkl"
    rgr = OnlineSVR(epsilon=1e-3, verbose=0)
    rgr.fit(X[:20], y[:20])
    dump(rgr, filename)
    del rgr

    loaded_rgr = load(filename)
    assert loaded_rgr._libosvr_.GetSamplesTrainedNumber() == 20

    loaded_rgr.partial_fit(X[20:-5], y[20:-5])
    assert loaded_rgr._libosvr_.GetSamplesTrainedNumber() == len(X[:-5])

    y_hat = loaded_rgr.predict(X[-5:])
    np.testing.assert_array_almost_equal(y[-5:], y_hat, decimal=2)
