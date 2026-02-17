#ifndef FUNCTIONS_H
#define FUNCTIONS_H

/**
 * Compute pairwise Pearson correlation coefficients between all row-pairs
 * of the input matrix.
 *
 * @param ny     Number of rows (vectors)
 * @param nx     Number of columns (elements per vector)
 * @param data   Input matrix stored row-major: data[x + y*nx]
 * @param result Output matrix (ny x ny), stored row-major.
 *               For all 0 <= j <= i < ny, result[i + j*ny] holds the
 *               Pearson correlation between row i and row j.
 */
void correlate(int ny, int nx, const float* data, float* result);

#endif // FUNCTIONS_H