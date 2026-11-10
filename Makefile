CXX ?= g++
MPICXX ?= mpicxx
NVCXX ?= nvc++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic
OMPFLAGS ?= -fopenmp
OPENMP_GPU_FLAGS ?= -O3 -std=c++17 -mp=gpu
INCLUDES := -Iinclude

.PHONY: all openmp_gpu clean test

all: bin/preprocess bin/generate_synthetic bin/pagerank_serial bin/pagerank_hybrid

bin:
	mkdir -p bin

bin/preprocess: src/preprocess.cpp include/common.hpp include/fast_input.hpp | bin
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@

bin/generate_synthetic: src/generate_synthetic.cpp include/common.hpp | bin
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@

bin/pagerank_serial: src/pagerank_serial.cpp include/common.hpp | bin
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@

bin/pagerank_hybrid: src/pagerank_hybrid.cpp include/common.hpp include/partition.hpp | bin
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) $(INCLUDES) $< -o $@

openmp_gpu: bin/pagerank_openmp_gpu

bin/pagerank_openmp_gpu: src/pagerank_openmp_gpu.cpp include/common.hpp | bin
	$(NVCXX) $(OPENMP_GPU_FLAGS) $(INCLUDES) $< -o $@

test: all
	bash scripts/run_correctness.sh

clean:
	rm -f bin/preprocess bin/generate_synthetic bin/pagerank_serial bin/pagerank_hybrid bin/pagerank_openmp_gpu
	rm -rf tests/generated
