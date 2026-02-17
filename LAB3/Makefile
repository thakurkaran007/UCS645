# ─────────────────────────────────────────────────────────────────────────────
#  Makefile  –  UCS645 Assignment: Pairwise Correlation
# ─────────────────────────────────────────────────────────────────────────────

# Compiler
CXX = g++

# Flags
#   -std=c++11  : C++11 standard
#   -Wall       : all warnings
#   -O3         : maximum optimisation (enables auto-vectorisation)
#   -march=native : use all CPU extensions (AVX2 / FMA if available)
#   -fopenmp    : OpenMP multi-threading
CXXFLAGS = -std=c++11 -Wall -O3 -march=native -fopenmp

# Executable name
TARGET = correlate

# Source / header files
SOURCES = main.cpp functions.cpp
HEADERS = functions.h
OBJECTS = $(SOURCES:.cpp=.o)

# ── Default target ────────────────────────────────────────────────────────────
all: $(TARGET)

# ── Link ──────────────────────────────────────────────────────────────────────
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# ── Compile each .cpp → .o ────────────────────────────────────────────────────
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# ── Run with a small matrix (quick smoke-test) ────────────────────────────────
run: $(TARGET)
	./$(TARGET) 64 128

# ── Benchmark targets ─────────────────────────────────────────────────────────
# Usage: make bench NY=500 NX=1000 THREADS=4
NY      ?= 500
NX      ?= 1000
THREADS ?= $(shell nproc)

bench: $(TARGET)
	@echo "=== Sequential (1 thread) ==="
	./$(TARGET) $(NY) $(NX) 1
	@echo ""
	@echo "=== Parallel ($(THREADS) threads) ==="
	./$(TARGET) $(NY) $(NX) $(THREADS)

# Perf-stat wrapper (requires Linux perf tool)
# Usage: make perf_seq NY=500 NX=1000
perf_seq: $(TARGET)
	perf stat -e cycles,instructions,cache-misses,cache-references \
	    ./$(TARGET) $(NY) $(NX) 1

perf_par: $(TARGET)
	perf stat -e cycles,instructions,cache-misses,cache-references \
	    ./$(TARGET) $(NY) $(NX) $(THREADS)

# Scaling experiment: 1 → max threads, fixed matrix size
scale: $(TARGET)
	@for t in $$(seq 1 $(THREADS)); do \
	    echo -n "threads=$$t  "; \
	    ./$(TARGET) $(NY) $(NX) $$t | grep "wall time"; \
	done

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	rm -f $(OBJECTS) $(TARGET)

# ── Phony targets ─────────────────────────────────────────────────────────────
.PHONY: all run bench perf_seq perf_par scale clean