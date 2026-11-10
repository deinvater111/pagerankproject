# MPI-OpenMP PageRank on the Pokec Social Network

This repository contains the source code and Vanda batch scripts used for the
DSA5208 Project 3 experiments. It implements:

- a serial C++ PageRank reference;
- a hybrid Message Passing Interface (MPI) and Open Multi-Processing (OpenMP)
  implementation;
- vertex-balanced and incoming-edge-balanced graph partitioning;
- serial, strong-scaling, weak-scaling, and MPI/OpenMP configuration studies;
- an OpenMP target-offload implementation for one NVIDIA GPU; and
- correctness checks and Pokec network analysis.

The repository is source-only. Downloaded graph data, compiled executables,
runtime CSV files, PBS logs, generated figures, and report files are intentionally
excluded.

## Quick start on Vanda

Run every command from the repository root. The PBS scripts use
`PBS_O_WORKDIR` to find the source code, data, and helper scripts.

```bash
git clone https://github.com/deinvater111/pagerankproject.git
cd pagerankproject
mkdir -p data
```

Download and decompress the Pokec relationship file as described below so that
`data/soc-pokec-relationships.txt` exists. Then load the CPU environment and
compile:

```bash
source config/vanda_env.sh
make -j 4
```

Submit preprocessing and wait for it to finish successfully:

```bash
qsub scripts/vanda_preprocess.pbs
```

Verify that preprocessing produced the graph metadata:

```bash
test -f data/pokec_graph/graph.meta && echo "Pokec preprocessing passed"
```

Run correctness validation before collecting performance measurements:

```bash
qsub scripts/vanda_correctness.pbs
```

After the correctness job finishes with `Exit_status = 0`, submit the performance
experiments in the order documented below.

## Directory structure

```text
Makefile                         Build rules
include/                         Shared C++ headers
src/preprocess.cpp               Convert an edge list to incoming CSR
src/generate_synthetic.cpp       Generate deterministic synthetic graphs
src/pagerank_serial.cpp          Serial PageRank reference
src/pagerank_hybrid.cpp          Hybrid MPI-OpenMP PageRank
src/pagerank_openmp_gpu.cpp      OpenMP target-offload PageRank
config/vanda_env.sh              Vanda GCC and Open MPI environment
config/vanda_gpu_env.sh          Vanda NVIDIA OpenMP-offload environment
config/results_manifest.conf     Curated CSV selection for final analysis
scripts/                         Validation, experiment, and analysis scripts
tests/                           Small directed graphs for correctness tests
```

## Dataset

Download the SNAP Pokec relationship network from:

<https://snap.stanford.edu/data/soc-Pokec.html>

Create the data directory, move the downloaded archive into it, and decompress
the archive:

```bash
mkdir -p data
mv /path/to/soc-pokec-relationships.txt.gz data/
gunzip data/soc-pokec-relationships.txt.gz
```

The resulting file must be located at:

```text
data/soc-pokec-relationships.txt
```

Each input row `u v` represents a directed relationship from user identifier
`u` to user identifier `v`. The Pokec relationship dataset contains 1,632,803 vertices
and 30,622,564 directed edges. The dataset itself is not included in this
repository.

## Vanda software environment

CPU experiments were tested with:

```bash
module purge
module load gompi/2022a
```

This loads GCC 11.3.0, Open MPI 4.1.4, `g++`, `mpicxx`, and `mpiexec`.
Load the tested environment with:

```bash
source config/vanda_env.sh
```

The GPU experiment uses:

```bash
module load nvhpc-hpcx-cuda12/26.1
```

The corresponding environment is loaded automatically by
`scripts/vanda_openmp_gpu.pbs`.

## Build

From the repository root on Vanda:

```bash
source config/vanda_env.sh
make -j 4
```

This creates:

```text
bin/preprocess
bin/generate_synthetic
bin/pagerank_serial
bin/pagerank_hybrid
```

The manual build verifies that the compiler and MPI environment are available.
The preprocessing PBS job performs a clean CPU rebuild on its allocated compute
node before converting the Pokec graph.

The OpenMP GPU executable is compiled on a GPU node by the GPU PBS job. It can
also be built after loading the NVIDIA environment:

```bash
source config/vanda_gpu_env.sh
make openmp_gpu
```

## Preprocess Pokec

Create the data directory, place the downloaded file there, and submit:

```bash
mkdir -p data
qsub scripts/vanda_preprocess.pbs
```

The job creates an incoming Compressed Sparse Row (CSR) graph under
`data/pokec_graph/`. PageRank timing excludes this preprocessing stage.

## Command-line programs

The following commands are functional examples for an allocated compute node or
a local MPI environment. Do not run computational experiments directly on a
Vanda login node. Use the supplied PBS scripts for the reported Vanda timings.

Run the serial implementation directly with:

```bash
bin/pagerank_serial \
  --graph data/pokec_graph \
  --fixed-iterations 100 \
  --top-k 20
```

Run a small hybrid example with:

```bash
export OMP_NUM_THREADS=2
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=cores

mpiexec -np 2 bin/pagerank_hybrid \
  --graph data/pokec_graph \
  --partition vertices \
  --fixed-iterations 100 \
  --top-k 20
```

Important options are:

| Option | Meaning |
|---|---|
| `--graph DIR` | Preprocessed graph directory |
| `--partition vertices\|edges` | MPI partitioning method |
| `--damping VALUE` | Damping factor; default `0.85` |
| `--tolerance VALUE` | L1 convergence tolerance |
| `--max-iterations N` | Maximum convergence iterations |
| `--fixed-iterations N` | Run exactly `N` iterations |
| `--top-k N` | Print the highest-ranked original vertex IDs |
| `--output FILE` | Append a summary row to a CSV file |
| `--scores-output FILE` | Write the full score vector as CSV |
| `--scores-binary-output FILE` | Write the full score vector in binary form |
| `--quiet` | Suppress ordinary terminal output |

The performance experiments use exactly 100 PageRank iterations and do not
write full score vectors during timed runs.

## Correctness validation

Submit the validation job before collecting performance measurements:

```bash
qsub scripts/vanda_correctness.pbs
```

It performs three levels of validation:

1. Serial and hybrid tests on cycle, star, dangling, and disconnected graphs.
2. Full-vector comparison on a deterministic graph with 100,000 vertices and
   1.6 million edges.
3. Serial convergence and top-vertex checks on the full Pokec graph.

The CPU hybrid score vector is compared with the serial reference using L1 and
maximum absolute differences. GPU validation uses the same comparison with
floating-point tolerances.

After completion, verify the PBS status and inspect the comparison CSV:

```bash
qstat -xf <JOBID>.stdct-mgmt-02 | grep -E \
"job_state|Exit_status|resources_used.walltime|resources_used.cput"

cat results/processed/score_vector_comparison_<JOBID>.stdct-mgmt-02.csv
```

A successful job has `job_state = F`, `Exit_status = 0`, and `passed=1` in the
comparison file.

## Performance experiments

Each performance script accepts a `REPETITIONS` variable. The reported results
used three measurements per configuration: a one-repetition pilot followed by
two additional repetitions.

For each timing script, submit a pilot first:

```bash
experiment=scripts/vanda_serial_benchmark.pbs
qsub -v REPETITIONS=1 "$experiment"
```

After validating the pilot, collect two additional measurements:

```bash
qsub -v REPETITIONS=2 "$experiment"
```

Replace the value of `experiment` with the timing script being measured.

Preparation jobs, correctness jobs, and placement smoke tests are run once and
do not require three repetitions.

### Serial Pokec baseline

```bash
qsub -v REPETITIONS=1 scripts/vanda_serial_benchmark.pbs
```

### Serial scaling with vertices and edges

```bash
qsub scripts/vanda_prepare_serial_scaling.pbs
```

Wait for graph preparation to finish successfully, then submit:

```bash
qsub -v REPETITIONS=1 scripts/vanda_serial_scaling.pbs
```

The first graph series varies vertices from 250,000 to 4,000,000 at fixed
out-degree 16. The second fixes 1,000,000 vertices and varies out-degree over
4, 8, 16, and 32.

### Pure-MPI strong scaling

One node, 1-72 ranks:

```bash
qsub -v REPETITIONS=1 scripts/vanda_strong_scaling.pbs
```

Two nodes, 144 ranks:

```bash
qsub scripts/vanda_multinode_smoke.pbs
```

The smoke-test log must end with `Two-node smoke test passed`. Then submit:

```bash
qsub -v REPETITIONS=1 scripts/vanda_multinode.pbs
```

Four nodes, 256 ranks:

```bash
qsub -v REPETITIONS=1 scripts/vanda_strong_256.pbs
```

Both vertex-balanced and edge-balanced partitioning are tested.

### Hybrid MPI-OpenMP configuration sweeps

Verify one-node placement first:

```bash
qsub scripts/vanda_hybrid_binding_smoke.pbs
```

The log must end with `Hybrid binding smoke test passed` before timings are
collected.

Run the 72-core one-node sweep:

```bash
qsub -v REPETITIONS=1 scripts/vanda_hybrid_sweep.pbs
```

Run the 144-core two-node sweep:

```bash
qsub scripts/vanda_multinode_smoke.pbs
```

After the multi-node smoke test passes:

```bash
qsub -v REPETITIONS=1 scripts/vanda_multinode_hybrid.pbs
```

### Weak scaling through 256 ranks

Generate the complete graph series:

```bash
qsub scripts/vanda_prepare_weak_256.pbs
```

After preparation completes, submit:

```bash
qsub -v REPETITIONS=1 scripts/vanda_weak_256_onenode.pbs
qsub -v REPETITIONS=1 scripts/vanda_weak_256_2node.pbs
qsub -v REPETITIONS=1 scripts/vanda_weak_256_4node.pbs
```

These jobs cover 1-64, 128, and 256 ranks. Each rank adds 100,000 destination
vertices and an average of 1.6 million edges. Because the generator deliberately
skews edge targets, actual incoming-edge counts differ among ranks.

### OpenMP GPU offload

```bash
qsub -v REPETITIONS=1 scripts/vanda_openmp_gpu.pbs
```

The job:

1. builds a same-node serial CPU reference;
2. compiles `pagerank_openmp_gpu` using `nvc++ -mp=gpu`;
3. sets `OMP_TARGET_OFFLOAD=MANDATORY`;
4. validates the complete GPU score vector against the serial reference; and
5. times both the serial CPU and GPU iteration loops on the same node.

The job log must report `device_offload=1`, its PBS history must report
`Exit_status = 0`, and the generated GPU score-comparison CSV must have
`passed=1`.

### Pokec data analysis

```bash
qsub scripts/vanda_data_analysis.pbs
```

This calculates the dangling-vertex fraction, PageRank score concentration,
PageRank/in-degree correlation, and top anonymous Pokec identifiers.

## Process placement on Vanda

Do not add Open MPI's `:PE=` mapping modifier. NUS HPC support confirmed that it
can cause multi-node launcher hangs on Vanda. The supplied launcher uses:

- `--bind-to core --map-by slot` for one-node pure MPI;
- `--bind-to core --map-by ppr:<ranks-per-node>:node` for multi-node pure MPI;
- unbound MPI ranks followed by `scripts/bind_openmp_rank.sh` for hybrid runs.

For hybrid runs, `OMP_PROC_BIND=close` and `OMP_PLACES=cores` place threads inside
the disjoint CPU set assigned to each local rank. Run the supplied smoke tests
before collecting timings.

## Outputs and monitoring

PBS scripts create output directories as needed:

```text
results/raw/         Per-job timing CSV files
results/processed/   Validation and data-analysis outputs
figures/             Optional generated figures
```

Monitor jobs with:

```bash
qstat -u "$USER"
```

A submitted job continues after the SSH connection closes. When it finishes,
check that its PBS history reports `Exit_status = 0` and inspect the final log
message and CSV row count.

Use the complete job identifier printed by `qsub`:

```bash
qstat -xf <JOBID>.stdct-mgmt-02 | grep -E \
"job_state|Exit_status|resources_used.walltime|resources_used.cput|resources_used.mem"
```

A successful completed job reports:

```text
job_state = F
Exit_status = 0
```

## Post-processing

The final R analysis script requires R with `dplyr`, `ggplot2`, `readr`,
`scales`, and `tidyr`. It reads the selected result filenames from
`config/results_manifest.conf`. Paths in the manifest are relative to the
repository root, multiple files are comma-separated, and shell-style wildcards
are supported. Exact filenames are preferred because they prevent failed pilots
or obsolete jobs from entering the analysis accidentally.

The committed manifest records the Vanda files used for the submitted report.
For a new experiment series, copy it and replace the filenames with the new job
IDs:

```bash
cp config/results_manifest.conf config/results_manifest.local.conf
```

Run the analysis with the committed manifest:

```bash
Rscript scripts/analyze_and_plot_results_v4.R \
  "$PWD" \
  analysis_output \
  config/results_manifest.conf
```

For newly collected results, pass the edited local manifest instead:

```bash
Rscript scripts/analyze_and_plot_results_v4.R \
  "$PWD" \
  analysis_output \
  config/results_manifest.local.conf
```

The script validates numerical invariants and repetition counts before writing
figures and summary tables to `analysis_output/`.

The raw timing files are not included in this source-only submission package.

## Reproducing the reported measurements

For each performance experiment:

1. submit a one-repetition pilot;
2. confirm successful completion and numerical invariants;
3. submit the same script with `REPETITIONS=2`;
4. combine all three measurements;
5. report the median, minimum, and maximum; and
6. calculate strong-scaling speedup and efficiency from the matching one-rank
   MPI baseline.

The PageRank implementations write graph size, iteration count, residual, rank
sum, minimum score, setup time, total time, computation time, communication time,
and throughput to their result CSV files.
