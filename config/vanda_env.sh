#!/usr/bin/env bash

module purge
module load gompi/2022a

if ! command -v g++ >/dev/null 2>&1; then
  echo "g++ is unavailable." >&2
  return 2
fi

if ! command -v mpicxx >/dev/null 2>&1; then
  echo "mpicxx is unavailable." >&2
  return 2
fi

if ! command -v mpiexec >/dev/null 2>&1; then
  echo "mpiexec is unavailable." >&2
  return 2
fi