#!/bin/bash

set -x

project="${1:-unset}"
if [[ ${project} == unset ]]; then
     project="${PWD}"
fi

mv ${project}/pyonlinesvr ${project}/pyonlinesvr.bak
pytest ${project}/tests
exit_code=$?
mv ${project}/pyonlinesvr.bak ${project}/pyonlinesvr
exit ${exit_code}
