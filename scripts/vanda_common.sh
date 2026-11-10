#!/usr/bin/env bash

set -euo pipefail

project_dir="${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${project_dir}"

source "${project_dir}/config/vanda_env.sh"

export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=cores

mkdir -p "${project_dir}/results/raw" "${project_dir}/results/processed" "${project_dir}/figures"

run_mpi() {
  local ranks="$1"
  local threads="$2"
  shift 2
  export OMP_NUM_THREADS="${threads}"

  # Vanda support confirmed that Open MPI's :PE= mapping can make daemons exit.
  # Pure MPI uses ordinary core binding. Hybrid processes are launched unbound
  # and the wrapper assigns each local rank a disjoint subset of the PBS cpuset.
  if [[ -n "${PBS_NODEFILE:-}" && -f "${PBS_NODEFILE}" ]]; then
    local allocated_nodes
    allocated_nodes="$(sort -u "${PBS_NODEFILE}" | wc -l | tr -d ' ')"
    if (( allocated_nodes > 1 )); then
      echo "run_mpi is restricted to one-node jobs; use run_mpi_ppr for multi-node execution." >&2
      return 2
    fi
  fi

  if mpiexec --version 2>&1 | grep -Eqi "Open MPI|OpenRTE"; then
    local mpi_prefix
    mpi_prefix="${EBROOTOPENMPI:-$(dirname "$(dirname "$(command -v mpiexec)")")}" 
    if (( threads == 1 )); then
      mpiexec --prefix "${mpi_prefix}" -np "${ranks}" \
        --bind-to core --map-by slot "$@"
    else
      mpiexec --prefix "${mpi_prefix}" -np "${ranks}" \
        --bind-to none --map-by slot \
        "${project_dir}/scripts/bind_openmp_rank.sh" "$@"
    fi
  else
    # Intel MPI and other launchers use different binding flags. PBS still
    # controls the allocated cores; verify placement in a pilot run.
    mpiexec -np "${ranks}" "$@"
  fi
}

run_mpi_ppr() {
  local ranks="$1"
  local ranks_per_node="$2"
  local threads="$3"
  shift 3
  export OMP_NUM_THREADS="${threads}"

  if mpiexec --version 2>&1 | grep -Eqi "Open MPI|OpenRTE"; then
    local mpi_prefix
    mpi_prefix="${EBROOTOPENMPI:-$(dirname "$(dirname "$(command -v mpiexec)")")}" 
    if (( threads == 1 )); then
      mpiexec --prefix "${mpi_prefix}" -np "${ranks}" \
        --bind-to core \
        --map-by "ppr:${ranks_per_node}:node" \
        "$@"
    else
      mpiexec --prefix "${mpi_prefix}" -np "${ranks}" \
        --bind-to none \
        --map-by "ppr:${ranks_per_node}:node" \
        "${project_dir}/scripts/bind_openmp_rank.sh" "$@"
    fi
  else
    echo "Explicit ranks-per-node placement requires Open MPI/OpenRTE." >&2
    return 2
  fi
}
