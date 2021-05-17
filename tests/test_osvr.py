import numpy as np
from pyonlinesvr import OnlineSVR
from joblib import load, dump
from pathlib import Path
from sklearn.utils.estimator_checks import check_estimator

a = np.sin(np.arange(0, 8, 0.1))
X = a.reshape(-1, 1)[:-1]
y = np.roll(a, -1)[:-1]
del a


def test_estimator():
    check_estimator(OnlineSVR())


def test_init():
    rgr = OnlineSVR()
    assert rgr is not None
    assert rgr.C == 30.0
    assert rgr.epsilon == 0.1
    assert rgr.kernel == "rbf"
    assert rgr.degree == 3
    assert rgr.gamma == None
    assert rgr.coef0 == 0.0
    assert rgr.tol == 1e-3
    assert rgr.stabilized == True
    assert rgr.save_kernel_matrix == True
    assert rgr.verbose == 0


def test_fit():
    rgr = OnlineSVR(C=0.1, verbose=0)
    rgr.fit(X, y)

    assert hasattr(
        rgr,
        "_libosvr_",
    )
    assert hasattr(rgr, "_shape_fit_")
    assert rgr._shape_fit_ == X.shape
    assert rgr._libosvr_.GetC() == 0.1
    assert rgr._libosvr_.GetVerbosity() == 0
    assert rgr._libosvr_.GetSamplesTrainedNumber() == len(X)


def test_partial_fit_begin():
    rgr = OnlineSVR(C=0.1, verbose=0)
    rgr.partial_fit(X, y)

    assert hasattr(
        rgr,
        "_libosvr_",
    )
    assert hasattr(rgr, "_shape_fit_")
    assert rgr._shape_fit_ == X.shape
    assert rgr._libosvr_.GetSamplesTrainedNumber() == len(X)


def test_partial_fit_continue():
    rgr = OnlineSVR(C=0.1, verbose=0)
    rgr.fit(X[:5], y[:5])

    assert hasattr(
        rgr,
        "_libosvr_",
    )
    assert hasattr(rgr, "_shape_fit_")
    assert rgr._shape_fit_ == (5, X.shape[1])
    assert rgr._libosvr_.GetSamplesTrainedNumber() == 5

    rgr.partial_fit(X[5:], y[5:])
    assert rgr._shape_fit_[1] == X.shape[1]
    assert rgr._libosvr_.GetSamplesTrainedNumber() == len(X)


def test_predict():
    rgr = OnlineSVR(epsilon=1e-3, verbose=0)
    rgr.fit(X[:-5], y[:-5])
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
