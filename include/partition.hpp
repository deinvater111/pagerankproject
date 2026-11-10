#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace pr {

inline std::vector<std::uint64_t> make_partitions(
    const std::vector<std::uint64_t>& row_offsets,
    int processes,
    const std::string& method) {
    if (processes <= 0 || row_offsets.empty()) throw std::runtime_error("Invalid partition input");
    const std::uint64_t vertices = row_offsets.size() - 1;
    const std::uint64_t edges = row_offsets.back();
    std::vector<std::uint64_t> boundaries(static_cast<std::size_t>(processes) + 1, 0);
    boundaries.back() = vertices;

    for (int r = 1; r < processes; ++r) {
        if (method == "vertices") {
            boundaries[static_cast<std::size_t>(r)] =
                (vertices * static_cast<std::uint64_t>(r)) / static_cast<std::uint64_t>(processes);
        } else if (method == "edges") {
            const std::uint64_t target =
                (edges * static_cast<std::uint64_t>(r)) / static_cast<std::uint64_t>(processes);
            auto it = std::lower_bound(row_offsets.begin(), row_offsets.end(), target);
            boundaries[static_cast<std::size_t>(r)] =
                static_cast<std::uint64_t>(std::distance(row_offsets.begin(), it));
            boundaries[static_cast<std::size_t>(r)] =
                std::min(boundaries[static_cast<std::size_t>(r)], vertices);
        } else {
            throw std::runtime_error("Unknown partition method: " + method);
        }
        boundaries[static_cast<std::size_t>(r)] =
            std::max(boundaries[static_cast<std::size_t>(r)], boundaries[static_cast<std::size_t>(r - 1)]);
    }
    return boundaries;
}

}  // namespace pr
