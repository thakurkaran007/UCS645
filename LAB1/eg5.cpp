#pragma omp parallel for collapse(2)
for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        // Simple math instead of rand() to avoid thread-locking
        A[i][j] = (double)(i + j); 
        B[i][j] = (double)(i * j);
    }
}