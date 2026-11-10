#include "common.hpp"
#include "fast_input.hpp"

#include <cstdio>

namespace {

struct Arguments {
    std::string input;
    std::string output;
};

Arguments parse_arguments(int argc, char** argv) {
    const auto options = pr::parse_options(argc, argv);
    return {pr::require_option(options, "--input"), pr::require_option(options, "--output")};
}

template <typename Function>
std::uint64_t scan_edges(const std::string& path, Function&& function) {
    pr::FastInput input(path);
    std::uint64_t source = 0;
    std::uint64_t destination = 0;
    std::uint64_t count = 0;
    while (input.read_uint64(source)) {
        if (!input.read_uint64(destination)) {
            throw std::runtime_error("Input contains an incomplete final edge");
        }
        function(source, destination);
        ++count;
    }
    return count;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto args = parse_arguments(argc, argv);
        const double start = pr::wall_seconds();

        std::cerr << "Pass 1/3: discovering vertices...\n";
        std::vector<std::uint8_t> present(1, 0);
        std::uint64_t maximum_id = 0;
        const std::uint64_t raw_edges = scan_edges(args.input, [&](std::uint64_t source, std::uint64_t destination) {
            const std::uint64_t required = std::max(source, destination) + 1;
            if (required > present.size()) present.resize(static_cast<std::size_t>(required), 0);
            present[static_cast<std::size_t>(source)] = 1;
            present[static_cast<std::size_t>(destination)] = 1;
            maximum_id = std::max(maximum_id, std::max(source, destination));
        });

        const std::uint64_t vertices =
            static_cast<std::uint64_t>(std::count(present.begin(), present.end(), std::uint8_t{1}));
        if (vertices == 0 || vertices > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("Unsupported vertex count");
        }

        const auto unmapped = std::numeric_limits<std::uint32_t>::max();
        std::vector<std::uint32_t> dense_id(present.size(), unmapped);
        std::vector<std::uint64_t> original_ids;
        original_ids.reserve(static_cast<std::size_t>(vertices));
        for (std::uint64_t id = 0; id < present.size(); ++id) {
            if (present[static_cast<std::size_t>(id)] != 0) {
                dense_id[static_cast<std::size_t>(id)] = static_cast<std::uint32_t>(original_ids.size());
                original_ids.push_back(id);
            }
        }
        present.clear();
        present.shrink_to_fit();

        std::cerr << "Pass 2/3: counting incoming and outgoing relationships...\n";
        std::vector<std::uint64_t> indegree(static_cast<std::size_t>(vertices), 0);
        std::vector<std::uint32_t> outdegree(static_cast<std::size_t>(vertices), 0);
        std::uint64_t self_edges = 0;
        const std::uint64_t edges = scan_edges(args.input, [&](std::uint64_t source, std::uint64_t destination) {
            const auto u = dense_id.at(static_cast<std::size_t>(source));
            const auto v = dense_id.at(static_cast<std::size_t>(destination));
            if (u == unmapped || v == unmapped) throw std::runtime_error("Internal ID mapping error");
            ++indegree[v];
            if (outdegree[u] == std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error("A vertex outdegree exceeds uint32 capacity");
            }
            ++outdegree[u];
            if (u == v) ++self_edges;
        });
        if (edges != raw_edges) throw std::runtime_error("Edge count changed between passes");

        std::vector<std::uint64_t> row_offsets(static_cast<std::size_t>(vertices) + 1, 0);
        for (std::size_t i = 0; i < indegree.size(); ++i) {
            row_offsets[i + 1] = row_offsets[i] + indegree[i];
        }
        std::vector<std::uint64_t> cursor = row_offsets;
        std::vector<std::uint32_t> sources(static_cast<std::size_t>(edges));

        std::cerr << "Pass 3/3: building incoming CSR graph...\n";
        const std::uint64_t filled = scan_edges(args.input, [&](std::uint64_t source, std::uint64_t destination) {
            const auto u = dense_id.at(static_cast<std::size_t>(source));
            const auto v = dense_id.at(static_cast<std::size_t>(destination));
            sources[static_cast<std::size_t>(cursor[v]++)] = u;
        });
        if (filled != edges) throw std::runtime_error("Edge count changed during CSR construction");

        const std::filesystem::path output_dir(args.output);
        std::filesystem::create_directories(output_dir);
        {
            std::ofstream meta(output_dir / "graph.meta", std::ios::trunc);
            if (!meta) throw std::runtime_error("Cannot create graph.meta");
            meta << "PAGERANK_GRAPH_V1 " << vertices << ' ' << edges << '\n';
        }
        pr::write_binary_vector(output_dir / "row_offsets.bin", row_offsets);
        pr::write_binary_vector(output_dir / "sources.bin", sources);
        pr::write_binary_vector(output_dir / "outdegree.bin", outdegree);
        pr::write_binary_vector(output_dir / "original_ids.bin", original_ids);

        const auto dangling = static_cast<std::uint64_t>(
            std::count(outdegree.begin(), outdegree.end(), std::uint32_t{0}));
        const auto no_incoming = static_cast<std::uint64_t>(
            std::count(indegree.begin(), indegree.end(), std::uint64_t{0}));
        const auto max_incoming = *std::max_element(indegree.begin(), indegree.end());
        const auto max_outgoing = *std::max_element(outdegree.begin(), outdegree.end());

        std::ofstream stats(output_dir / "preprocess_stats.txt", std::ios::trunc);
        stats << "vertices=" << vertices << '\n'
              << "edges=" << edges << '\n'
              << "maximum_original_id=" << maximum_id << '\n'
              << "self_edges=" << self_edges << '\n'
              << "dangling_vertices=" << dangling << '\n'
              << "vertices_without_incoming_edges=" << no_incoming << '\n'
              << "maximum_indegree=" << max_incoming << '\n'
              << "maximum_outdegree=" << max_outgoing << '\n'
              << "preprocess_seconds=" << std::setprecision(17) << (pr::wall_seconds() - start) << '\n';

        std::cout << "Preprocessing complete\n"
                  << "vertices=" << vertices << " edges=" << edges << '\n'
                  << "dangling_vertices=" << dangling << " self_edges=" << self_edges << '\n'
                  << "output=" << output_dir << '\n'
                  << "seconds=" << std::setprecision(6) << (pr::wall_seconds() - start) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "preprocess: " << error.what() << '\n';
        return 1;
    }
}
