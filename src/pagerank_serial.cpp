#include "common.hpp"

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
        const double setup_seconds = pr::wall_seconds() - setup_start;

        std::vector<double> rank(static_cast<std::size_t>(meta.vertices), 1.0 / meta.vertices);
        std::vector<double> next_rank(static_cast<std::size_t>(meta.vertices), 0.0);
        std::vector<double> contribution(static_cast<std::size_t>(meta.vertices), 0.0);

        double dangling_mass = 0.0;
        for (std::uint64_t v = 0; v < meta.vertices; ++v) {
            if (outdegree[static_cast<std::size_t>(v)] == 0) dangling_mass += rank[static_cast<std::size_t>(v)];
        }

        const int iteration_limit = config.fixed_iterations > 0 ? config.fixed_iterations : config.max_iterations;
        double residual = std::numeric_limits<double>::infinity();
        bool converged = false;
        int completed_iterations = 0;
        const double start = pr::wall_seconds();

        for (int iteration = 0; iteration < iteration_limit; ++iteration) {
            for (std::uint64_t v = 0; v < meta.vertices; ++v) {
                const auto degree = outdegree[static_cast<std::size_t>(v)];
                contribution[static_cast<std::size_t>(v)] =
                    degree == 0 ? 0.0 : rank[static_cast<std::size_t>(v)] / degree;
            }

            const double base = (1.0 - config.damping) / meta.vertices
                                + config.damping * dangling_mass / meta.vertices;
            residual = 0.0;
            double next_dangling = 0.0;
            for (std::uint64_t v = 0; v < meta.vertices; ++v) {
                double incoming_sum = 0.0;
                for (std::uint64_t edge = row_offsets[static_cast<std::size_t>(v)];
                     edge < row_offsets[static_cast<std::size_t>(v + 1)]; ++edge) {
                    incoming_sum += contribution[sources[static_cast<std::size_t>(edge)]];
                }
                const double value = base + config.damping * incoming_sum;
                next_rank[static_cast<std::size_t>(v)] = value;
                residual += std::abs(value - rank[static_cast<std::size_t>(v)]);
                if (outdegree[static_cast<std::size_t>(v)] == 0) next_dangling += value;
            }
            rank.swap(next_rank);
            dangling_mass = next_dangling;
            completed_iterations = iteration + 1;
            if (config.fixed_iterations == 0 && residual < config.tolerance) {
                converged = true;
                break;
            }
        }
        const double total_seconds = pr::wall_seconds() - start;
        if (config.fixed_iterations > 0) converged = residual < config.tolerance;

        const double rank_sum = std::accumulate(rank.begin(), rank.end(), 0.0);
        const double min_rank = *std::min_element(rank.begin(), rank.end());

        pr::RunResult result;
        result.program = "serial";
        result.dataset = graph_dir.filename().string();
        result.vertices = meta.vertices;
        result.edges = meta.edges;
        result.partition = "serial";
        result.schedule = "serial";
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
            std::cout << "program=serial vertices=" << meta.vertices << " edges=" << meta.edges
                      << " iterations=" << completed_iterations << " residual=" << std::setprecision(17)
                      << residual << " rank_sum=" << rank_sum << " total_seconds=" << total_seconds << '\n';
            pr::print_top_k(rank, original_ids, config.top_k);
        }
        pr::write_scores_csv(config.scores_output, rank, original_ids, row_offsets, outdegree);
        pr::write_scores_binary(config.scores_binary_output, rank);
        pr::append_result(config.output_csv, result);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "pagerank_serial: " << error.what() << '\n';
        return 1;
    }
}
