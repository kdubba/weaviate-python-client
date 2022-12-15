#!/bin/bash

set -eo pipefail

for i in {1..1000}; do
    python3 replication_test_import.py > import.log
    echo "iteration ${i} complete"
    sleep 1
done
