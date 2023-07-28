#!/bin/bash

set -e

mv {project}/pyonlinesvr {project}/pyonlinesvr.bak
pytest {project}/tests
exit_code=$?
mv {project}/pyonlinesvr.bak {project}/pyonlinesvr
exit $exit_code
