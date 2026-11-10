#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
input_path="${1:-${project_dir}/data/soc-pokec-relationships.txt}"
output_dir="${2:-${project_dir}/data/pokec_graph}"

if [[ ! -f "${input_path}" ]]; then
  echo "Input file not found: ${input_path}" >&2
  exit 1
fi

"${project_dir}/bin/preprocess" --input "${input_path}" --output "${output_dir}"
echo "Prepared graph: ${output_dir}"
