#include <stdio.h>
#include <omp.h>

int main() {

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        printf("Thread %d before barrier\n", tid);

        #pragma omp barrier

        printf("Thread %d after barrier\n", tid);
    }

    return 0;
}
