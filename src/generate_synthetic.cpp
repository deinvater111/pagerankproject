#include "common.hpp"

#include <array>

namespace {

std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27U)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31U);
}

struct Arguments {
    std::uint64_t vertices = 0;
    std::uint32_t edges_per_vertex = 16;
    std::uint64_t seed = 5208;
    std::string model = "skewed";
    std::string output;
};

Arguments parse_arguments(int argc, char** argv) {
    const auto options = pr::parse_options(argc, argv);
    Arguments args;
    args.vertices = std::stoull(pr::require_option(options, "--vertices"));
    args.output = pr::require_option(options, "--output");
    if (auto it = options.find("--edges-per-vertex"); it != options.end()) {
        args.edges_per_vertex = static_cast<std::uint32_t>(std::stoul(it->second));
    }
    if (auto it = options.find("--seed"); it != options.end()) args.seed = std::stoull(it->second);
    if (auto it = options.find("--model"); it != options.end()) args.model = it->second;
    if (args.vertices < 2 || args.vertices > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("--vertices must be between 2 and uint32 maximum");
    }
    if (args.edges_per_vertex == 0 || args.edges_per_vertex >= args.vertices) {
        throw std::runtime_error("--edges-per-vertex must be positive and less than vertex count");
    }
    if (args.model != "uniform" && args.model != "skewed") {
        throw std::runtime_error("--model must be uniform or skewed");
    }
    return args;
}

std::uint32_t candidate_target(const Arguments& args, std::uint64_t source, std::uint64_t attempt) {
    const std::uint64_t random = splitmix64(args.seed ^ (source * 0x9e3779b97f4a7c15ULL) ^ attempt);
    if (args.model == "skewed" && (random % 10U) < 8U) {
        const std::uint64_t hub_count = std::max<std::uint64_t>(1, args.vertices / 10U);
        return static_cast<std::uint32_t>((random >> 8U) % hub_count);
    }
    return static_cast<std::uint32_t>((random >> 8U) % args.vertices);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto args = parse_arguments(argc, argv);
        const double start = pr::wall_seconds();
        const std::uint64_t edges = args.vertices * args.edges_per_vertex;
        std::vector<std::uint64_t> indegree(static_cast<std::size_t>(args.vertices), 0);

        std::cerr << "Counting synthetic incoming edges...\n";
        for (std::uint64_t source = 0; source < args.vertices; ++source) {
            std::vector<std::uint32_t> selected;
            selected.reserve(args.edges_per_vertex);
            std::uint64_t attempt = 0;
            while (selected.size() < args.edges_per_vertex) {
                const auto target = candidate_target(args, source, attempt++);
                if (target == source || std::find(selected.begin(), selected.end(), target) != selected.end()) continue;
                selected.push_back(target);
                ++indegree[target];
            }
        }

        std::vector<std::uint64_t> row_offsets(static_cast<std::size_t>(args.vertices) + 1, 0);
        for (std::size_t i = 0; i < indegree.size(); ++i) row_offsets[i + 1] = row_offsets[i] + indegree[i];
        std::vector<std::uint64_t> cursor = row_offsets;
        std::vector<std::uint32_t> sources(static_cast<std::size_t>(edges));

        std::cerr << "Filling synthetic CSR graph...\n";
        for (std::uint64_t source = 0; source < args.vertices; ++source) {
            std::vector<std::uint32_t> selected;
            selected.reserve(args.edges_per_vertex);
            std::uint64_t attempt = 0;
            while (selected.size() < args.edges_per_vertex) {
                const auto target = candidate_target(args, source, attempt++);
                if (target == source || std::find(selected.begin(), selected.end(), target) != selected.end()) continue;
                selected.push_back(target);
                sources[static_cast<std::size_t>(cursor[target]++)] = static_cast<std::uint32_t>(source);
            }
        }

        std::vector<std::uint32_t> outdegree(static_cast<std::size_t>(args.vertices), args.edges_per_vertex);
        std::vector<std::uint64_t> original_ids(static_cast<std::size_t>(args.vertices));
        std::iota(original_ids.begin(), original_ids.end(), 0ULL);

        const std::filesystem::path output_dir(args.output);
        std::filesystem::create_directories(output_dir);
        std::ofstream meta(output_dir / "graph.meta", std::ios::trunc);
        meta << "PAGERANK_GRAPH_V1 " << args.vertices << ' ' << edges << '\n';
        meta.close();
        pr::write_binary_vector(output_dir / "row_offsets.bin", row_offsets);
        pr::write_binary_vector(output_dir / "sources.bin", sources);
        pr::write_binary_vector(output_dir / "outdegree.bin", outdegree);
        pr::write_binary_vector(output_dir / "original_ids.bin", original_ids);

        std::ofstream stats(output_dir / "preprocess_stats.txt", std::ios::trunc);
        stats << "generator=synthetic\nmodel=" << args.model << '\n'
              << "vertices=" << args.vertices << '\n' << "edges=" << edges << '\n'
              << "edges_per_vertex=" << args.edges_per_vertex << '\n'
              << "seed=" << args.seed << '\n'
              << "self_edges_rejected=1\n"
              << "duplicate_edges_rejected=1\n";
        if (args.model == "skewed") {
            stats << "hub_vertex_fraction=0.1\n"
                  << "hub_candidate_probability=0.8\n";
        }
        stats
              << "generation_seconds=" << std::setprecision(17) << (pr::wall_seconds() - start) << '\n';

        std::cout << "Generated " << args.model << " graph with " << args.vertices
                  << " vertices and " << edges << " edges in "
                  << std::setprecision(6) << (pr::wall_seconds() - start) << " seconds\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "generate_synthetic: " << error.what() << '\n';
        return 1;
    }
}
