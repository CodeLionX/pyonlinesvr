# PyOnlineSVR

[![pipeline status](https://github.com/CodeLionX/pyonlinesvr/actions/workflows/conda-python-test.yml/badge.svg)](https://github.com/CodeLionX/pyonlinesvr/actions/workflows/conda-python-test.yml)
![coverage report](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/CodeLionX/6762bee806477c52e079f21d2f252688/raw/pyonlinesvr__heads_main.json)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI package](https://badge.fury.io/py/PyOnlineSVR.svg)](https://badge.fury.io/py/PyOnlineSVR)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![python version 3.7|3.8|3.9](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue)

Python-Wrapper for Francesco Parrella's OnlineSVR [[PAR2007]](#PAR2007) C++ implementation with [scikit-learn](https://sklearn.org/)-compatible interfaces.
You can find more information about the OnlineSVR [here](http://onlinesvr.altervista.org/) and the original source code [here](https://github.com/fp2556/onlinesvr/tree/master/c%2B%2B).

## Installation

### Dependencies

PyOnlineSVR requires the following dependencies:

- python (>=3.7)
- numpy (>=1.13.3)
- scipy (>=0.19.1)
- joblib (>=0.11)
- scikit-learn (>=0.23.0)

### Binaries

PyOnlineSVR is published to [PyPi](https://pypi.org/project/PyOnlineSVR/) and can be installed using `pip`.

#### Prerequisites

- python (>=3.7)
- pip (>=19.0 to support [manylinux2010](https://github.com/pypa/manylinux))

#### Steps

You can use `pip` to install PyOnlineSVR using:

```sh
pip install PyOnlineSVR
```

### From Source (Linux)

If you are installing PyOnlineSVR from source, you will need Python 3.7 or later and a modern C++ compiler.
We highly recommend using an [Anaconda](https://www.anaconda.com/products/individual#download-section) environment for building this project.

In the following, we explain the steps to build PyOnlineSVR using **Anaconda** and **git**.

#### Prepare environment

Create a new Anaconda environment and install the required dependencies.
This includes python, [SWIG](http://swig.org/) to generate the C++ wrapper, and the C and C++ compiler toolchains.

```bash
conda create -n pyonlinesvr python swig gcc_linux-64 gxx_linux-64
conda activate pyonlinesvr
```

#### Install dependencies

```bash
conda install -n pyonlinesvr numpy scipy scikit-learn
```

#### Get the source code

```bash
git clone https://github.com/CodeLionX/pyonlinesvr.git
cd pyonlinesvr
```

#### Install PyOnlineSVR

```bash
python setup.py install
```

Note that if your are using Anaconda, you may experience an error caused by the linker:

```txt
build/temp.linux-x86_64-3.7/torch/csrc/stub.o: file not recognized: file format not recognized
collect2: error: ld returned 1 exit status
error: command 'g++' failed with exit status 1
```

This is caused by the linker `ld` from the Conda environment shadowing the system `ld`.
You should use a newer version of Python in your environment that fixes this issue.
The recommended Python versions are (3.6.10+,) 3.7.6+ and 3.8.1+.
For further details see [the issue](https://github.com/ContinuumIO/anaconda-issues/issues/11152).

## Usage

```python
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
```

## License

PyOnlineSVR is free software under the terms of the GNU General Public License, as found in the [LICENSE](./LICENSE) file.

## References

<a name="PAR2007">[PAR2007]</a>: Parrelly, Francesco (2007). "Online Support Vector Machines for Regression." Master thesis. University of Genoa, Italy. [PDF](http://onlinesvr.altervista.org/Research/Online%20Support%20Vector%20Regression%20(Parrella%20F.)%20[2007].pdf)
