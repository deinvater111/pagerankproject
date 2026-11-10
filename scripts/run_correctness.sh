#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
generated_dir="${project_dir}/tests/generated"
result_file="${project_dir}/results/processed/correctness.csv"

mkdir -p "${generated_dir}" "$(dirname "${result_file}")"
: > "${result_file}"

for graph in cycle star dangling disconnected; do
  "${project_dir}/bin/preprocess" \
    --input "${project_dir}/tests/${graph}.txt" \
    --output "${generated_dir}/${graph}"

  "${project_dir}/bin/pagerank_serial" \
    --graph "${generated_dir}/${graph}" \
    --tolerance 1e-12 \
    --max-iterations 500 \
    --top-k 5 \
    --output "${result_file}"

  if command -v mpiexec >/dev/null 2>&1 && [[ -x "${project_dir}/bin/pagerank_hybrid" ]]; then
    OMP_NUM_THREADS=2 OMP_PROC_BIND=close OMP_PLACES=cores \
      mpiexec -np 2 "${project_dir}/bin/pagerank_hybrid" \
        --graph "${generated_dir}/${graph}" \
        --partition vertices \
        --schedule static \
        --tolerance 1e-12 \
        --max-iterations 500 \
        --top-k 5 \
        --output "${result_file}"
  fi
done

echo "Correctness runs completed: ${result_file}"
