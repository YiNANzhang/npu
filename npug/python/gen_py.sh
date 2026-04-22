#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
flatc --python -o generated ../schema/npug.fbs
echo "generated Python bindings in python/generated/"
