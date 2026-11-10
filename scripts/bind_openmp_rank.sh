#!/usr/bin/env bash

# Restrict one MPI process to a disjoint slice of the CPUs granted by PBS.
# Open MPI launches this wrapper with --bind-to none.  GNU OpenMP then sees
# only the selected CPUs and OMP_PLACES=cores pins its threads inside that set.

set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "Usage: bind_openmp_rank.sh PROGRAM [ARGUMENT ...]" >&2
  exit 2
fi

taskset_command="${TASKSET_COMMAND:-taskset}"
if ! command -v "${taskset_command}" >/dev/null 2>&1; then
  echo "${taskset_command} is required for manual hybrid-rank binding." >&2
  exit 2
fi

threads="${OMP_NUM_THREADS:?OMP_NUM_THREADS is not set}"
local_rank="${OMPI_COMM_WORLD_LOCAL_RANK:?OMPI_COMM_WORLD_LOCAL_RANK is not set}"

if ! [[ "${threads}" =~ ^[1-9][0-9]*$ && "${local_rank}" =~ ^[0-9]+$ ]]; then
  echo "Invalid OMP_NUM_THREADS=${threads} or OMPI_COMM_WORLD_LOCAL_RANK=${local_rank}." >&2
  exit 2
fi

allowed_list="${ALLOWED_CPU_LIST_OVERRIDE:-$(awk '/^Cpus_allowed_list:/ {print $2}' /proc/self/status)}"
if [[ -z "${allowed_list}" ]]; then
  echo "Could not determine the CPUs allowed by PBS." >&2
  exit 2
fi

cpus=()
IFS=',' read -ra cpu_ranges <<< "${allowed_list}"
for cpu_range in "${cpu_ranges[@]}"; do
  if [[ "${cpu_range}" == *-* ]]; then
    first="${cpu_range%-*}"
    last="${cpu_range#*-}"
    for ((cpu=first; cpu<=last; cpu++)); do
      cpus+=("${cpu}")
    done
  else
    cpus+=("${cpu_range}")
  fi
done

start=$((local_rank * threads))
end=$((start + threads))
if (( end > ${#cpus[@]} )); then
  echo "Rank ${local_rank} needs CPUs [${start},${end}), but PBS allowed only ${#cpus[@]} CPUs (${allowed_list})." >&2
  exit 2
fi

rank_cpus=()
for ((index=start; index<end; index++)); do
  rank_cpus+=("${cpus[index]}")
done
cpu_csv="$(IFS=,; echo "${rank_cpus[*]}")"

if [[ "${REPORT_RANK_BINDING:-0}" == "1" ]]; then
  echo "host=$(hostname) local_rank=${local_rank} threads=${threads} cpus=${cpu_csv}" >&2
fi

exec "${taskset_command}" --cpu-list "${cpu_csv}" "$@"
