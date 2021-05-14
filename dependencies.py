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
import argparse

# adapted from https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/_min_dependencies.py

# dependency name -> (version spec, comma-seperated tags)
dependent_packages = {
    "joblib": ("0.11", "install"),
    "numpy": ("1.13.3", "build, install, test"),
    "scipy": ("0.19.1", "build, install, test"),
    "scikit-learn": ("0.23.0", "build, install, test"),
    "flake8": ("3.9.2", "test"),
    "mypy": ("0.812", "test"),
    "twine": ("3.4.1", "deploy"),
    "pytest": ("6.2.4", "test"),
    "pytest-cov": ("2.11.1", "test"),
}

# inverse mapping: tag -> dependency-specs (compatible to setuptools)
packages_for_tag = {tag: [] for tag in ["build", "install", "test", "doc", "deploy"]}
for package, (min_version, tags) in dependent_packages.items():
    for tag in tags.split(","):
        packages_for_tag[tag.strip()].append(f"{package}>={min_version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get min version for packages")

    parser.add_argument("--package", choices=dependent_packages.keys(), required=False)
    parser.add_argument("--tag", choices=packages_for_tag.keys(), required=False)
    args = parser.parse_args()

    if args.package and args.tag:
        exit(-1)
    if not args.package and args.tag:
        pkgs = " ".join(packages_for_tag[args.tag])
        pkgs = pkgs.replace(">", "=")
        print(pkgs)
    elif args.package and not args.tag:
        min_version = dependent_packages[args.package][0]
        print(min_version)
    else:
        all = []
        for p in dependent_packages:
            all.append(f"{p}=={dependent_packages[p][0]}")
        print(" ".join(all))