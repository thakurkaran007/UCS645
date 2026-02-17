#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <omp.h>
#include "functions.h"

// ─────────────────────────────────────────────────────────────────────────────
//  Usage:
//    ./correlate <ny> <nx> [num_threads]
//
//  ny            = number of rows (vectors)
//  nx            = number of columns (elements per vector)
//  num_threads   = OpenMP thread count (optional, default = max available)
//
//  Timing is printed to stdout; use  perf stat ./correlate ...  to collect
//  hardware-performance-counter data alongside it.
// ─────────────────────────────────────────────────────────────────────────────

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <ny> <nx> [num_threads]\n"
              << "  ny           number of rows  (vectors)\n"
              << "  nx           number of columns (elements per vector)\n"
              << "  num_threads  OpenMP thread count (default: system max)\n";
}

// Simple pseudo-random fill so results are reproducible
static void fill_matrix(int ny, int nx, std::vector<float>& mat) {
    mat.resize((size_t)ny * nx);
    unsigned seed = 42;
    for (int i = 0; i < ny * nx; ++i) {
        seed = seed * 1664525u + 1013904223u;   // LCG
        mat[i] = (float)(int(seed & 0xFFFF) - 32768) / 32768.0f;
    }
}

// Pretty-print a duration
static void print_elapsed(const char* label,
                           std::chrono::high_resolution_clock::time_point t0,
                           std::chrono::high_resolution_clock::time_point t1)
{
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << label << ": " << ms << " ms\n";
}

// Spot-check a few results against a naive reference (only for small matrices)
static bool verify(int ny, int nx,
                   const std::vector<float>& data,
                   const std::vector<float>& result)
{
    // Reference: naive double-precision correlate for a small subset
    const int CHECK = std::min(ny, 8);
    for (int i = 0; i < CHECK; ++i) {
        for (int j = 0; j <= i; ++j) {
            // compute mean_i, mean_j
            double si = 0, sj = 0;
            for (int x = 0; x < nx; ++x) {
                si += data[x + i * nx];
                sj += data[x + j * nx];
            }
            double mi = si / nx, mj = sj / nx;

            double num = 0, di2 = 0, dj2 = 0;
            for (int x = 0; x < nx; ++x) {
                double ai = data[x + i * nx] - mi;
                double aj = data[x + j * nx] - mj;
                num += ai * aj;
                di2 += ai * ai;
                dj2 += aj * aj;
            }
            double denom = std::sqrt(di2 * dj2);
            float ref = (denom > 0) ? (float)(num / denom) : 0.0f;
            float got = result[i + j * ny];
            if (std::fabs(ref - got) > 1e-4f) {
                std::cerr << "VERIFY FAIL at (" << i << "," << j << "): "
                          << "ref=" << ref << " got=" << got << "\n";
                return false;
            }
        }
    }
    return true;
}

int main(int argc, char* argv[])
{
    // ── Parse arguments ───────────────────────────────────────────────────────
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    int ny = std::atoi(argv[1]);
    int nx = std::atoi(argv[2]);
    int num_threads = (argc >= 4) ? std::atoi(argv[3]) : omp_get_max_threads();

    if (ny <= 0 || nx <= 0 || num_threads <= 0) {
        std::cerr << "Error: ny, nx, and num_threads must be positive integers.\n";
        print_usage(argv[0]);
        return 1;
    }

    omp_set_num_threads(num_threads);

    std::cout << "──────────────────────────────────────────\n"
              << " Matrix correlation benchmark\n"
              << "──────────────────────────────────────────\n"
              << " ny           = " << ny          << "\n"
              << " nx           = " << nx          << "\n"
              << " num_threads  = " << num_threads << "\n"
              << " result cells = " << (long long)ny * (ny + 1) / 2 << "\n"
              << "──────────────────────────────────────────\n";

    // ── Allocate & fill input matrix ─────────────────────────────────────────
    std::vector<float> data, result((size_t)ny * ny, 0.0f);
    fill_matrix(ny, nx, data);

    // ── Run & time correlate() ────────────────────────────────────────────────
    auto t0 = std::chrono::high_resolution_clock::now();
    correlate(ny, nx, data.data(), result.data());
    auto t1 = std::chrono::high_resolution_clock::now();

    print_elapsed(" correlate() wall time", t0, t1);

    // ── Verify (only practical for small matrices) ────────────────────────────
    if (ny <= 512 && nx <= 512) {
        if (verify(ny, nx, data, result))
            std::cout << " Verification: PASSED\n";
        else
            std::cout << " Verification: FAILED\n";
    } else {
        std::cout << " Verification: skipped (matrix too large)\n";
    }

    // ── Print a small corner of the result for sanity ─────────────────────────
    std::cout << " result[0,0] (should be 1.0) = " << result[0] << "\n";
    if (ny > 1)
        std::cout << " result[1,0]               = " << result[1] << "\n";
    if (ny > 2)
        std::cout << " result[2,1]               = " << result[2 + 1*ny] << "\n";

    std::cout << "──────────────────────────────────────────\n";
    return 0;
}