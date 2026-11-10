#!/usr/bin/env bash

module purge
module load nvhpc-hpcx-cuda12/26.1

if ! command -v nvc++ >/dev/null 2>&1; then
  echo "nvc++ is unavailable after loading nvhpc-hpcx-cuda12/26.1." >&2
  return 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable on this node." >&2
  return 2
fi
