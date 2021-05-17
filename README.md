# PyOnlineSVR

[![pipeline status](https://gitlab.hpi.de/akita/pyonlinesvr/badges/main/pipeline.svg)](https://gitlab.hpi.de/akita/pyonlinesvr/-/commits/main)
[![coverage report](https://gitlab.hpi.de/akita/pyonlinesvr/badges/main/coverage.svg)](https://gitlab.hpi.de/akita/pyonlinesvr/-/commits/main)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![release info](https://img.shields.io/badge/Release-0.0.1-blue)](https://gitlab.hpi.de/akita/bp2020fn1/timeeval/-/releases/v0.0.1)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![python version 3.6|3.7|3.8|3.9](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue)](#)

Python-Wrapper for Francesco Parrella's OnlineSVR [[PAR2007]](#PAR2007) C++ implementation with [scikit-learn](https://sklearn.org/)-compatible interfaces.
You can find more information about the OnlineSVR [here](http://onlinesvr.altervista.org/) and the original source code [here](https://github.com/fp2556/onlinesvr/tree/master/c%2B%2B).

## Installation

### Dependencies

PyOnlineSVR requires the following dependencies:

- python (>=3.6.10)
- numpy (>=1.13.3)
- scipy (>=0.19.1)
- joblib (>=0.11)
- scikit-learn (>=0.23.0)

### Binaries

The binaries of PyOnlineSVR are published to the [internal package registry](https://gitlab.hpi.de/akita/pyonlinesvr/-/packages) of the Gitlab instance running at [gitlab.hpi.de](https://gitlab.hpi.de/).

#### Prerequisites

- python (>=3.6.10)
- pip
- A [personal access token](https://gitlab.hpi.de/help/user/profile/personal_access_tokens.md) with the scope set to `api` (read) or another type of access token able to read the package registry hosted at [gitlab.hpi.de](https://gitlab.hpi.de/).

#### Steps

You can use `pip` to install PyOnlineSVR using:

```sh
pip install PyOnlineSVR --extra-index-url https://__token__:<your_personal_token>@gitlab.hpi.de/api/v4/projects/4434/packages/pypi/simple
```

### From Source

If you are installing PyOnlineSVR from source, you will need Python 3.6.10 or later and a modern C++ compiler.
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
git clone https://gitlab.hpi.de/akita/pyonlinesvr
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
The recommended Python versions are 3.6.10+, 3.7.6+ and 3.8.1+.
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
