#include "functions.h"
#include <cmath>
#include <vector>
#include <omp.h>
#include <immintrin.h>   // AVX / SSE intrinsics

// ─────────────────────────────────────────────────────────────────────────────
//  INTERNAL HELPERS
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Normalise each row of `data` so that it has zero mean and unit length,
 * storing the result in `norm` (double precision).
 * Also used by all three implementation levels.
 */
static void normalise_rows(int ny, int nx,
                            const float*  data,
                            std::vector<double>& norm)
{
    norm.resize((size_t)ny * nx);

#pragma omp parallel for schedule(static)
    for (int y = 0; y < ny; ++y) {
        // compute mean
        double sum = 0.0;
        for (int x = 0; x < nx; ++x)
            sum += data[x + y * nx];
        double mean = sum / nx;

        // subtract mean
        double sq = 0.0;
        for (int x = 0; x < nx; ++x) {
            double v = data[x + y * nx] - mean;
            norm[x + y * nx] = v;
            sq += v * v;
        }

        // divide by L2-norm (guard against zero-variance rows)
        double inv = (sq > 0.0) ? 1.0 / std::sqrt(sq) : 0.0;
        for (int x = 0; x < nx; ++x)
            norm[x + y * nx] *= inv;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  TASK 1 — Sequential baseline (double precision throughout)
// ─────────────────────────────────────────────────────────────────────────────
static void correlate_sequential(int ny, int nx,
                                  const float* data,
                                  float*       result)
{
    // Step 1: normalise rows
    std::vector<double> norm;
    normalise_rows(ny, nx, data, norm);   // sequential-safe (single thread)

    // Step 2: for each lower-triangular pair (i, j), dot-product gives r
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j <= i; ++j) {
            double dot = 0.0;
            for (int x = 0; x < nx; ++x)
                dot += norm[x + i * nx] * norm[x + j * nx];
            // clamp to [-1, 1] to absorb floating-point drift
            if (dot >  1.0) dot =  1.0;
            if (dot < -1.0) dot = -1.0;
            result[i + j * ny] = (float)dot;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  TASK 2 — OpenMP multi-threaded (outer loop parallelised)
// ─────────────────────────────────────────────────────────────────────────────
static void correlate_openmp(int ny, int nx,
                              const float* data,
                              float*       result)
{
    std::vector<double> norm;
    normalise_rows(ny, nx, data, norm);

#pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j <= i; ++j) {
            double dot = 0.0;
            for (int x = 0; x < nx; ++x)
                dot += norm[x + i * nx] * norm[x + j * nx];
            if (dot >  1.0) dot =  1.0;
            if (dot < -1.0) dot = -1.0;
            result[i + j * ny] = (float)dot;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  TASK 3 — OpenMP + AVX2 vectorised inner dot-product
//           Falls back to scalar if AVX2 is unavailable at compile time.
// ─────────────────────────────────────────────────────────────────────────────

#ifdef __AVX2__

// Horizontal sum of a 256-bit AVX double register (4 × double)
static inline double hsum_avx(__m256d v) {
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d s  = _mm_add_pd(lo, hi);
    return _mm_cvtsd_f64(_mm_hadd_pd(s, s));
}

static double dot_avx(const double* a, const double* b, int n) {
    __m256d acc = _mm256_setzero_pd();
    int x = 0;
    for (; x <= n - 4; x += 4) {
        __m256d va = _mm256_loadu_pd(a + x);
        __m256d vb = _mm256_loadu_pd(b + x);
        acc = _mm256_fmadd_pd(va, vb, acc);   // FMA if available
    }
    double dot = hsum_avx(acc);
    for (; x < n; ++x)                         // handle tail
        dot += a[x] * b[x];
    return dot;
}

#endif // __AVX2__

static void correlate_vectorised(int ny, int nx,
                                  const float* data,
                                  float*       result)
{
    std::vector<double> norm;
    normalise_rows(ny, nx, data, norm);

#pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < ny; ++i) {
        const double* ri = &norm[i * nx];
        for (int j = 0; j <= i; ++j) {
            const double* rj = &norm[j * nx];
            double dot = 0.0;

#ifdef __AVX2__
            dot = dot_avx(ri, rj, nx);
#else
            for (int x = 0; x < nx; ++x)
                dot += ri[x] * rj[x];
#endif
            if (dot >  1.0) dot =  1.0;
            if (dot < -1.0) dot = -1.0;
            result[i + j * ny] = (float)dot;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  PUBLIC ENTRY POINT
//  Dispatches to the fastest available implementation (Task 3).
//  To test sequential (Task 1) or OpenMP-only (Task 2), change the call below.
// ─────────────────────────────────────────────────────────────────────────────
void correlate(int ny, int nx, const float* data, float* result)
{
    correlate_vectorised(ny, nx, data, result);

    // Uncomment one of the lines below to test the other variants:
    // correlate_sequential(ny, nx, data, result);
    // correlate_openmp(ny, nx, data, result);
}