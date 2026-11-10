#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace pr {

struct GraphMeta {
    std::uint64_t vertices = 0;
    std::uint64_t edges = 0;
};

struct RunConfig {
    std::string graph_dir;
    std::string output_csv;
    std::string scores_output;
    std::string scores_binary_output;
    std::string partition = "vertices";
    std::string schedule = "static";
    double damping = 0.85;
    double tolerance = 1e-10;
    int max_iterations = 100;
    int fixed_iterations = 0;
    int chunk_size = 64;
    int top_k = 10;
    bool quiet = false;
};

struct RunResult {
    std::string program;
    std::string dataset;
    std::uint64_t vertices = 0;
    std::uint64_t edges = 0;
    std::string partition;
    int mpi_processes = 1;
    int omp_threads = 1;
    std::string schedule = "serial";
    int iterations = 0;
    bool converged = false;
    double residual = 0.0;
    double rank_sum = 0.0;
    double min_rank = 0.0;
    double setup_seconds = 0.0;
    double total_seconds = 0.0;
    double compute_seconds = 0.0;
    double communication_seconds = 0.0;
    double allgather_seconds = 0.0;
    double allreduce_seconds = 0.0;
    double edges_per_second = 0.0;
    std::uint64_t local_vertices_min = 0;
    std::uint64_t local_vertices_max = 0;
    std::uint64_t local_edges_min = 0;
    std::uint64_t local_edges_max = 0;
};

inline std::map<std::string, std::string> parse_options(int argc, char** argv) {
    std::map<std::string, std::string> options;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key.rfind("--", 0) != 0) {
            throw std::runtime_error("Unexpected positional argument: " + key);
        }
        if (key == "--quiet") {
            options[key] = "true";
            continue;
        }
        if (i + 1 >= argc) {
            throw std::runtime_error("Missing value for " + key);
        }
        options[key] = argv[++i];
    }
    return options;
}

inline std::string require_option(const std::map<std::string, std::string>& options,
                                  const std::string& key) {
    auto it = options.find(key);
    if (it == options.end()) {
        throw std::runtime_error("Required option is missing: " + key);
    }
    return it->second;
}

inline RunConfig parse_run_config(int argc, char** argv) {
    const auto options = parse_options(argc, argv);
    RunConfig config;
    config.graph_dir = require_option(options, "--graph");
    if (auto it = options.find("--output"); it != options.end()) config.output_csv = it->second;
    if (auto it = options.find("--scores-output"); it != options.end()) config.scores_output = it->second;
    if (auto it = options.find("--scores-binary-output"); it != options.end()) {
        config.scores_binary_output = it->second;
    }
    if (auto it = options.find("--partition"); it != options.end()) config.partition = it->second;
    if (auto it = options.find("--schedule"); it != options.end()) config.schedule = it->second;
    if (auto it = options.find("--damping"); it != options.end()) config.damping = std::stod(it->second);
    if (auto it = options.find("--tolerance"); it != options.end()) config.tolerance = std::stod(it->second);
    if (auto it = options.find("--max-iterations"); it != options.end()) config.max_iterations = std::stoi(it->second);
    if (auto it = options.find("--fixed-iterations"); it != options.end()) config.fixed_iterations = std::stoi(it->second);
    if (auto it = options.find("--chunk-size"); it != options.end()) config.chunk_size = std::stoi(it->second);
    if (auto it = options.find("--top-k"); it != options.end()) config.top_k = std::stoi(it->second);
    config.quiet = options.count("--quiet") != 0;

    if (config.damping <= 0.0 || config.damping >= 1.0) {
        throw std::runtime_error("--damping must be between 0 and 1");
    }
    if (config.tolerance <= 0.0) throw std::runtime_error("--tolerance must be positive");
    if (config.max_iterations <= 0) throw std::runtime_error("--max-iterations must be positive");
    if (config.fixed_iterations < 0) throw std::runtime_error("--fixed-iterations cannot be negative");
    if (config.chunk_size <= 0) throw std::runtime_error("--chunk-size must be positive");
    if (config.partition != "vertices" && config.partition != "edges") {
        throw std::runtime_error("--partition must be vertices or edges");
    }
    if (config.schedule != "static" && config.schedule != "dynamic" && config.schedule != "guided") {
        throw std::runtime_error("--schedule must be static, dynamic, or guided");
    }
    return config;
}

inline GraphMeta read_meta(const std::filesystem::path& graph_dir) {
    std::ifstream input(graph_dir / "graph.meta");
    if (!input) throw std::runtime_error("Cannot open graph.meta in " + graph_dir.string());
    std::string magic;
    GraphMeta meta;
    input >> magic >> meta.vertices >> meta.edges;
    if (!input || magic != "PAGERANK_GRAPH_V1") {
        throw std::runtime_error("Invalid graph metadata in " + graph_dir.string());
    }
    return meta;
}

template <typename T>
inline std::vector<T> read_binary_vector(const std::filesystem::path& path,
                                         std::uint64_t count,
                                         std::uint64_t offset_elements = 0) {
    if (count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("Requested binary vector is too large");
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open " + path.string());
    input.seekg(static_cast<std::streamoff>(offset_elements * sizeof(T)), std::ios::beg);
    std::vector<T> data(static_cast<std::size_t>(count));
    if (count > 0) {
        input.read(reinterpret_cast<char*>(data.data()),
                   static_cast<std::streamsize>(count * sizeof(T)));
        if (!input) throw std::runtime_error("Failed reading " + path.string());
    }
    return data;
}

template <typename T>
inline void write_binary_vector(const std::filesystem::path& path, const std::vector<T>& data) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("Cannot create " + path.string());
    if (!data.empty()) {
        output.write(reinterpret_cast<const char*>(data.data()),
                     static_cast<std::streamsize>(data.size() * sizeof(T)));
    }
    if (!output) throw std::runtime_error("Failed writing " + path.string());
}

inline std::string csv_header() {
    return "program,dataset,vertices,edges,partition,mpi_processes,omp_threads,schedule,"
           "iterations,converged,residual,rank_sum,min_rank,setup_seconds,total_seconds,"
           "compute_seconds,communication_seconds,allgather_seconds,allreduce_seconds,"
           "edges_per_second,local_vertices_min,local_vertices_max,local_edges_min,local_edges_max";
}

inline std::string csv_row(const RunResult& r) {
    std::ostringstream out;
    out << std::setprecision(17)
        << r.program << ',' << r.dataset << ',' << r.vertices << ',' << r.edges << ','
        << r.partition << ',' << r.mpi_processes << ',' << r.omp_threads << ',' << r.schedule << ','
        << r.iterations << ',' << (r.converged ? 1 : 0) << ',' << r.residual << ','
        << r.rank_sum << ',' << r.min_rank << ',' << r.setup_seconds << ',' << r.total_seconds << ','
        << r.compute_seconds << ',' << r.communication_seconds << ',' << r.allgather_seconds << ','
        << r.allreduce_seconds << ',' << r.edges_per_second << ',' << r.local_vertices_min << ','
        << r.local_vertices_max << ',' << r.local_edges_min << ',' << r.local_edges_max;
    return out.str();
}

inline void append_result(const std::string& output_path, const RunResult& result) {
    if (output_path.empty()) return;
    const std::filesystem::path path(output_path);
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    const bool needs_header = !std::filesystem::exists(path) || std::filesystem::file_size(path) == 0;
    std::ofstream output(path, std::ios::app);
    if (!output) throw std::runtime_error("Cannot open result file " + output_path);
    if (needs_header) output << csv_header() << '\n';
    output << csv_row(result) << '\n';
}

inline void write_scores_csv(const std::string& output_path,
                             const std::vector<double>& ranks,
                             const std::vector<std::uint64_t>& original_ids,
                             const std::vector<std::uint64_t>& row_offsets,
                             const std::vector<std::uint32_t>& outdegree) {
    if (output_path.empty()) return;
    if (ranks.size() != original_ids.size() || ranks.size() != outdegree.size()
        || row_offsets.size() != ranks.size() + 1) {
        throw std::runtime_error("Cannot write scores: graph vectors have inconsistent sizes");
    }

    const std::filesystem::path path(output_path);
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::trunc);
    if (!output) throw std::runtime_error("Cannot create score file " + output_path);
    output << "dense_id,original_id,pagerank,indegree,outdegree\n";
    output << std::setprecision(17);
    for (std::size_t vertex = 0; vertex < ranks.size(); ++vertex) {
        const auto indegree = row_offsets[vertex + 1] - row_offsets[vertex];
        output << vertex << ',' << original_ids[vertex] << ',' << ranks[vertex] << ','
               << indegree << ',' << outdegree[vertex] << '\n';
    }
    if (!output) throw std::runtime_error("Failed writing score file " + output_path);
}

inline void write_scores_binary(const std::string& output_path,
                                const std::vector<double>& ranks) {
    if (output_path.empty()) return;
    const std::filesystem::path path(output_path);
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    write_binary_vector(path, ranks);
}

inline void print_top_k(const std::vector<double>& ranks,
                        const std::vector<std::uint64_t>& original_ids,
                        int top_k) {
    if (top_k <= 0 || ranks.empty()) return;
    const int k = std::min<int>(top_k, static_cast<int>(ranks.size()));
    std::vector<std::uint32_t> indices(ranks.size());
    std::iota(indices.begin(), indices.end(), 0U);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [&](std::uint32_t a, std::uint32_t b) { return ranks[a] > ranks[b]; });
    std::cout << "top_rank,original_id,pagerank\n";
    for (int i = 0; i < k; ++i) {
        const auto index = indices[static_cast<std::size_t>(i)];
        std::cout << (i + 1) << ',' << original_ids[index] << ','
                  << std::setprecision(17) << ranks[index] << '\n';
    }
}

inline double wall_seconds() {
    using clock = std::chrono::steady_clock;
    static const auto epoch = clock::now();
    return std::chrono::duration<double>(clock::now() - epoch).count();
}

}  // namespace pr
