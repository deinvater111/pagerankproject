#include "common.hpp"
#include "partition.hpp"

#include <mpi.h>
#include <omp.h>

namespace {

void check_mpi(int code, const char* operation) {
    if (code == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(code, message, &length);
    throw std::runtime_error(std::string(operation) + ": " + std::string(message, length));
}

void configure_schedule(const std::string& schedule, int chunk) {
    if (schedule == "static") omp_set_schedule(omp_sched_static, chunk);
    else if (schedule == "dynamic") omp_set_schedule(omp_sched_dynamic, chunk);
    else if (schedule == "guided") omp_set_schedule(omp_sched_guided, chunk);
    else throw std::runtime_error("Unsupported OpenMP schedule: " + schedule);
}

}  // namespace

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank_id = 0;
    int process_count = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    try {
        if (provided < MPI_THREAD_FUNNELED) {
            throw std::runtime_error("MPI implementation did not provide MPI_THREAD_FUNNELED");
        }
        const auto config = pr::parse_run_config(argc, argv);
        configure_schedule(config.schedule, config.chunk_size);
        const int threads = omp_get_max_threads();
        const double setup_start = MPI_Wtime();
        const std::filesystem::path graph_dir(config.graph_dir);

        pr::GraphMeta meta;
        std::vector<std::uint64_t> row_offsets;
        std::vector<std::uint32_t> outdegree;
        if (rank_id == 0) {
            meta = pr::read_meta(graph_dir);
            row_offsets =
                pr::read_binary_vector<std::uint64_t>(graph_dir / "row_offsets.bin", meta.vertices + 1);
            outdegree =
                pr::read_binary_vector<std::uint32_t>(graph_dir / "outdegree.bin", meta.vertices);
        }
        check_mpi(MPI_Bcast(&meta.vertices, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD), "MPI_Bcast(vertices)");
        check_mpi(MPI_Bcast(&meta.edges, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD), "MPI_Bcast(edges)");
        if (meta.vertices > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("MPI_Allgatherv implementation requires vertex count <= INT_MAX");
        }
        if (rank_id != 0) {
            row_offsets.resize(static_cast<std::size_t>(meta.vertices) + 1);
            outdegree.resize(static_cast<std::size_t>(meta.vertices));
        }
        check_mpi(MPI_Bcast(row_offsets.data(), static_cast<int>(meta.vertices + 1), MPI_UINT64_T, 0, MPI_COMM_WORLD),
                  "MPI_Bcast(row_offsets)");
        check_mpi(MPI_Bcast(outdegree.data(), static_cast<int>(meta.vertices), MPI_UINT32_T, 0, MPI_COMM_WORLD),
                  "MPI_Bcast(outdegree)");

        const auto boundaries = pr::make_partitions(row_offsets, process_count, config.partition);
        const std::uint64_t vertex_begin = boundaries[static_cast<std::size_t>(rank_id)];
        const std::uint64_t vertex_end = boundaries[static_cast<std::size_t>(rank_id + 1)];
        const std::uint64_t local_vertices = vertex_end - vertex_begin;
        const std::uint64_t edge_begin = row_offsets[static_cast<std::size_t>(vertex_begin)];
        const std::uint64_t edge_end = row_offsets[static_cast<std::size_t>(vertex_end)];
        const std::uint64_t local_edges = edge_end - edge_begin;

        std::vector<std::uint64_t> local_offsets(static_cast<std::size_t>(local_vertices) + 1);
        for (std::uint64_t i = 0; i <= local_vertices; ++i) {
            local_offsets[static_cast<std::size_t>(i)] =
                row_offsets[static_cast<std::size_t>(vertex_begin + i)] - edge_begin;
        }
        auto local_sources = pr::read_binary_vector<std::uint32_t>(
            graph_dir / "sources.bin", local_edges, edge_begin);

        std::vector<int> counts(static_cast<std::size_t>(process_count));
        std::vector<int> displacements(static_cast<std::size_t>(process_count));
        for (int r = 0; r < process_count; ++r) {
            const auto count = boundaries[static_cast<std::size_t>(r + 1)] - boundaries[static_cast<std::size_t>(r)];
            if (count > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
                throw std::runtime_error("Local vertex count exceeds MPI int count capacity");
            }
            counts[static_cast<std::size_t>(r)] = static_cast<int>(count);
            displacements[static_cast<std::size_t>(r)] = static_cast<int>(boundaries[static_cast<std::size_t>(r)]);
        }

        std::vector<double> local_rank(static_cast<std::size_t>(local_vertices), 1.0 / meta.vertices);
        std::vector<double> local_next(static_cast<std::size_t>(local_vertices), 0.0);
        std::vector<double> local_contribution(static_cast<std::size_t>(local_vertices), 0.0);
        std::vector<double> global_contribution(static_cast<std::size_t>(meta.vertices), 0.0);

        double local_dangling = 0.0;
        #pragma omp parallel for reduction(+:local_dangling) schedule(runtime)
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(local_vertices); ++i) {
            if (outdegree[static_cast<std::size_t>(vertex_begin + i)] == 0) {
                local_dangling += local_rank[static_cast<std::size_t>(i)];
            }
        }
        double dangling_mass = 0.0;
        check_mpi(MPI_Allreduce(&local_dangling, &dangling_mass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD),
                  "MPI_Allreduce(initial dangling mass)");

        check_mpi(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier");
        const double setup_seconds_local = MPI_Wtime() - setup_start;
        const double total_start = MPI_Wtime();
        double compute_seconds_local = 0.0;
        double allgather_seconds_local = 0.0;
        double allreduce_seconds_local = 0.0;
        double residual = std::numeric_limits<double>::infinity();
        int completed_iterations = 0;
        bool converged = false;
        const int iteration_limit = config.fixed_iterations > 0 ? config.fixed_iterations : config.max_iterations;

        for (int iteration = 0; iteration < iteration_limit; ++iteration) {
            double phase_start = MPI_Wtime();
            #pragma omp parallel for schedule(runtime)
            for (std::int64_t i = 0; i < static_cast<std::int64_t>(local_vertices); ++i) {
                const auto degree = outdegree[static_cast<std::size_t>(vertex_begin + i)];
                local_contribution[static_cast<std::size_t>(i)] =
                    degree == 0 ? 0.0 : local_rank[static_cast<std::size_t>(i)] / degree;
            }
            compute_seconds_local += MPI_Wtime() - phase_start;

            phase_start = MPI_Wtime();
            check_mpi(MPI_Allgatherv(local_contribution.data(), static_cast<int>(local_vertices), MPI_DOUBLE,
                                     global_contribution.data(), counts.data(), displacements.data(), MPI_DOUBLE,
                                     MPI_COMM_WORLD),
                      "MPI_Allgatherv(contributions)");
            allgather_seconds_local += MPI_Wtime() - phase_start;

            const double base = (1.0 - config.damping) / meta.vertices
                                + config.damping * dangling_mass / meta.vertices;
            double local_error = 0.0;
            double next_local_dangling = 0.0;
            phase_start = MPI_Wtime();
            #pragma omp parallel for reduction(+:local_error,next_local_dangling) schedule(runtime)
            for (std::int64_t i = 0; i < static_cast<std::int64_t>(local_vertices); ++i) {
                double incoming_sum = 0.0;
                for (std::uint64_t edge = local_offsets[static_cast<std::size_t>(i)];
                     edge < local_offsets[static_cast<std::size_t>(i + 1)]; ++edge) {
                    incoming_sum += global_contribution[local_sources[static_cast<std::size_t>(edge)]];
                }
                const double value = base + config.damping * incoming_sum;
                local_next[static_cast<std::size_t>(i)] = value;
                local_error += std::abs(value - local_rank[static_cast<std::size_t>(i)]);
                if (outdegree[static_cast<std::size_t>(vertex_begin + i)] == 0) next_local_dangling += value;
            }
            compute_seconds_local += MPI_Wtime() - phase_start;

            double local_metrics[2] = {local_error, next_local_dangling};
            double global_metrics[2] = {0.0, 0.0};
            phase_start = MPI_Wtime();
            check_mpi(MPI_Allreduce(local_metrics, global_metrics, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD),
                      "MPI_Allreduce(iteration metrics)");
            allreduce_seconds_local += MPI_Wtime() - phase_start;
            residual = global_metrics[0];
            dangling_mass = global_metrics[1];
            local_rank.swap(local_next);
            completed_iterations = iteration + 1;
            if (config.fixed_iterations == 0 && residual < config.tolerance) {
                converged = true;
                break;
            }
        }

        const double total_seconds_local = MPI_Wtime() - total_start;
        if (config.fixed_iterations > 0) converged = residual < config.tolerance;
        double local_sum = std::accumulate(local_rank.begin(), local_rank.end(), 0.0);
        double local_min = local_rank.empty() ? std::numeric_limits<double>::infinity()
                                              : *std::min_element(local_rank.begin(), local_rank.end());
        double rank_sum = 0.0;
        double min_rank = 0.0;
        check_mpi(MPI_Reduce(&local_sum, &rank_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD),
                  "MPI_Reduce(rank sum)");
        check_mpi(MPI_Reduce(&local_min, &min_rank, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD),
                  "MPI_Reduce(min rank)");

        double local_times[5] = {setup_seconds_local, total_seconds_local, compute_seconds_local,
                                 allgather_seconds_local, allreduce_seconds_local};
        double max_times[5] = {0, 0, 0, 0, 0};
        check_mpi(MPI_Reduce(local_times, max_times, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD),
                  "MPI_Reduce(timings)");
        std::uint64_t local_sizes[2] = {local_vertices, local_edges};
        std::uint64_t min_sizes[2] = {0, 0};
        std::uint64_t max_sizes[2] = {0, 0};
        check_mpi(MPI_Reduce(local_sizes, min_sizes, 2, MPI_UINT64_T, MPI_MIN, 0, MPI_COMM_WORLD),
                  "MPI_Reduce(min sizes)");
        check_mpi(MPI_Reduce(local_sizes, max_sizes, 2, MPI_UINT64_T, MPI_MAX, 0, MPI_COMM_WORLD),
                  "MPI_Reduce(max sizes)");

        std::vector<double> full_rank;
        if (rank_id == 0) full_rank.resize(static_cast<std::size_t>(meta.vertices));
        check_mpi(MPI_Gatherv(local_rank.data(), static_cast<int>(local_vertices), MPI_DOUBLE,
                              rank_id == 0 ? full_rank.data() : nullptr, counts.data(), displacements.data(),
                              MPI_DOUBLE, 0, MPI_COMM_WORLD),
                  "MPI_Gatherv(final ranks)");

        if (rank_id == 0) {
            const auto original_ids = pr::read_binary_vector<std::uint64_t>(
                graph_dir / "original_ids.bin", meta.vertices);
            pr::RunResult result;
            result.program = "hybrid";
            result.dataset = graph_dir.filename().string();
            result.vertices = meta.vertices;
            result.edges = meta.edges;
            result.partition = config.partition;
            result.mpi_processes = process_count;
            result.omp_threads = threads;
            result.schedule = config.schedule;
            result.iterations = completed_iterations;
            result.converged = converged;
            result.residual = residual;
            result.rank_sum = rank_sum;
            result.min_rank = min_rank;
            result.setup_seconds = max_times[0];
            result.total_seconds = max_times[1];
            result.compute_seconds = max_times[2];
            result.allgather_seconds = max_times[3];
            result.allreduce_seconds = max_times[4];
            result.communication_seconds = max_times[3] + max_times[4];
            result.edges_per_second = max_times[1] > 0.0
                ? static_cast<double>(meta.edges) * completed_iterations / max_times[1] : 0.0;
            result.local_vertices_min = min_sizes[0];
            result.local_vertices_max = max_sizes[0];
            result.local_edges_min = min_sizes[1];
            result.local_edges_max = max_sizes[1];

            if (!config.quiet) {
                std::cout << "program=hybrid partition=" << config.partition
                          << " mpi_processes=" << process_count << " omp_threads=" << threads
                          << " iterations=" << completed_iterations << " residual=" << std::setprecision(17)
                          << residual << " rank_sum=" << rank_sum << " total_seconds=" << max_times[1]
                          << " compute_seconds=" << max_times[2]
                          << " communication_seconds=" << result.communication_seconds << '\n';
                pr::print_top_k(full_rank, original_ids, config.top_k);
            }
            pr::write_scores_csv(config.scores_output, full_rank, original_ids, row_offsets, outdegree);
            pr::write_scores_binary(config.scores_binary_output, full_rank);
            pr::append_result(config.output_csv, result);
        }

        MPI_Finalize();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "rank " << rank_id << " pagerank_hybrid: " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
