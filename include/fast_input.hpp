#pragma once

#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace pr {

class FastInput {
public:
    explicit FastInput(const std::string& path) : file_(std::fopen(path.c_str(), "rb")) {
        if (!file_) throw std::runtime_error("Cannot open input file: " + path);
    }

    ~FastInput() {
        if (file_) std::fclose(file_);
    }

    FastInput(const FastInput&) = delete;
    FastInput& operator=(const FastInput&) = delete;

    bool read_uint64(std::uint64_t& value) {
        int c = next_char();
        while (c != EOF && (c < '0' || c > '9')) c = next_char();
        if (c == EOF) return false;
        value = 0;
        do {
            value = value * 10 + static_cast<std::uint64_t>(c - '0');
            c = next_char();
        } while (c >= '0' && c <= '9');
        return true;
    }

private:
    static constexpr std::size_t buffer_size_ = 1U << 20;
    FILE* file_ = nullptr;
    char buffer_[buffer_size_];
    std::size_t position_ = 0;
    std::size_t available_ = 0;

    int next_char() {
        if (position_ == available_) {
            available_ = std::fread(buffer_, 1, buffer_size_, file_);
            position_ = 0;
            if (available_ == 0) return EOF;
        }
        return static_cast<unsigned char>(buffer_[position_++]);
    }
};

}  // namespace pr
