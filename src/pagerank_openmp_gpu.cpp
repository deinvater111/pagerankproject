#include "common.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char** argv) {
    try {
        const auto config = pr::parse_run_config(argc, argv);
        const double setup_start = pr::wall_seconds();
        const std::filesystem::path graph_dir(config.graph_dir);
        const auto meta = pr::read_meta(graph_dir);
        const auto row_offsets =
            pr::read_binary_vector<std::uint64_t>(graph_dir / "row_offsets.bin", meta.vertices + 1);
        const auto sources =
            pr::read_binary_vector<std::uint32_t>(graph_dir / "sources.bin", meta.edges);
        const auto outdegree =
            pr::read_binary_vector<std::uint32_t>(graph_dir / "outdegree.bin", meta.vertices);
        const auto original_ids =
            pr::read_binary_vector<std::uint64_t>(graph_dir / "original_ids.bin", meta.vertices);

        if (meta.vertices == 0) throw std::runtime_error("Graph has no vertices");

        std::vector<double> rank(static_cast<std::size_t>(meta.vertices), 1.0 / meta.vertices);
        std::vector<double> contribution(static_cast<std::size_t>(meta.vertices), 0.0);
        double dangling_mass = 0.0;
        for (std::uint64_t vertex = 0; vertex < meta.vertices; ++vertex) {
            if (outdegree[static_cast<std::size_t>(vertex)] == 0) {
                dangling_mass += rank[static_cast<std::size_t>(vertex)];
            }
        }

        const std::uint64_t vertices = meta.vertices;
        [[maybe_unused]] const std::uint64_t edges = meta.edges;
        const double damping = config.damping;
        const std::uint64_t* offsets_data = row_offsets.data();
        const std::uint32_t* sources_data = sources.data();
        const std::uint32_t* outdegree_data = outdegree.data();
        double* rank_data = rank.data();
        double* contribution_data = contribution.data();
        double residual = std::numeric_limits<double>::infinity();
        bool converged = false;
        int completed_iterations = 0;
        bool used_gpu = false;
        double setup_seconds = 0.0;
        double total_seconds = 0.0;
        const int iteration_limit = config.fixed_iterations > 0
            ? config.fixed_iterations : config.max_iterations;

        // Keep the graph, PageRank vector, and contribution vector resident on
        // the device across every PageRank iteration.  The update is safe in
        // place because all reads use the contribution vector computed from
        // the previous PageRank state.
        #pragma omp target data \
            map(to: offsets_data[0:vertices + 1], sources_data[0:edges], \
                    outdegree_data[0:vertices]) \
            map(tofrom: rank_data[0:vertices]) \
            map(alloc: contribution_data[0:vertices])
        {
            int initial_device = 1;
            #pragma omp target map(from: initial_device)
            {
                #ifdef _OPENMP
                initial_device = omp_is_initial_device();
                #else
                initial_device = 1;
                #endif
            }
            used_gpu = initial_device == 0;
            setup_seconds = pr::wall_seconds() - setup_start;
            const double total_start = pr::wall_seconds();

            for (int iteration = 0; iteration < iteration_limit; ++iteration) {
                #pragma omp target teams distribute parallel for
                for (std::uint64_t vertex = 0; vertex < vertices; ++vertex) {
                    const std::uint32_t degree = outdegree_data[vertex];
                    contribution_data[vertex] = degree == 0
                        ? 0.0 : rank_data[vertex] / static_cast<double>(degree);
                }

                const double base = (1.0 - damping) / vertices
                                    + damping * dangling_mass / vertices;
                residual = 0.0;
                double next_dangling = 0.0;
                #pragma omp target teams distribute parallel for \
                    reduction(+:residual,next_dangling)
                for (std::uint64_t vertex = 0; vertex < vertices; ++vertex) {
                    double incoming_sum = 0.0;
                    for (std::uint64_t edge = offsets_data[vertex];
                         edge < offsets_data[vertex + 1]; ++edge) {
                        incoming_sum += contribution_data[sources_data[edge]];
                    }
                    const double value = base + damping * incoming_sum;
                    residual += std::abs(value - rank_data[vertex]);
                    rank_data[vertex] = value;
                    if (outdegree_data[vertex] == 0) next_dangling += value;
                }

                dangling_mass = next_dangling;
                completed_iterations = iteration + 1;
                if (config.fixed_iterations == 0 && residual < config.tolerance) {
                    converged = true;
                    break;
                }
            }
            total_seconds = pr::wall_seconds() - total_start;
        }

#ifdef _OPENMP
        if (!used_gpu) {
            throw std::runtime_error(
                "OpenMP target region ran on the host instead of an accelerator");
        }
#endif
        if (config.fixed_iterations > 0) converged = residual < config.tolerance;

        const double rank_sum = std::accumulate(rank.begin(), rank.end(), 0.0);
        const double min_rank = *std::min_element(rank.begin(), rank.end());

        pr::RunResult result;
        result.program = "openmp_gpu";
        result.dataset = graph_dir.filename().string();
        result.vertices = meta.vertices;
        result.edges = meta.edges;
        result.partition = "gpu";
        result.schedule = "openmp_target";
        result.iterations = completed_iterations;
        result.converged = converged;
        result.residual = residual;
        result.rank_sum = rank_sum;
        result.min_rank = min_rank;
        result.setup_seconds = setup_seconds;
        result.total_seconds = total_seconds;
        result.compute_seconds = total_seconds;
        result.edges_per_second = total_seconds > 0.0
            ? static_cast<double>(meta.edges) * completed_iterations / total_seconds : 0.0;
        result.local_vertices_min = meta.vertices;
        result.local_vertices_max = meta.vertices;
        result.local_edges_min = meta.edges;
        result.local_edges_max = meta.edges;

        if (!config.quiet) {
            std::cout << "program=openmp_gpu device_offload=" << (used_gpu ? 1 : 0)
                      << " vertices=" << meta.vertices << " edges=" << meta.edges
                      << " iterations=" << completed_iterations
                      << " residual=" << std::setprecision(17) << residual
                      << " rank_sum=" << rank_sum
                      << " total_seconds=" << total_seconds << '\n';
            pr::print_top_k(rank, original_ids, config.top_k);
        }
        pr::write_scores_csv(config.scores_output, rank, original_ids, row_offsets, outdegree);
        pr::write_scores_binary(config.scores_binary_output, rank);
        pr::append_result(config.output_csv, result);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "pagerank_openmp_gpu: " << error.what() << '\n';
        return 1;
    }
}
