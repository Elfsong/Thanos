# %%
import measure
import statistics
from tqdm import tqdm

# %%
code = r"""
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 2000  // You can adjust this value to make the computation more intensive

void multiply_matrices(int **a, int **b, int **c, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            c[i][j] = 0;
            for (int k = 0; k < n; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

int main() {
    // Allocate and initialize matrices
    int **a = malloc(N * sizeof(int*));
    int **b = malloc(N * sizeof(int*));
    int **c = malloc(N * sizeof(int*));
    for (int i = 0; i < N; i++) {
        a[i] = malloc(N * sizeof(int));
        b[i] = malloc(N * sizeof(int));
        c[i] = malloc(N * sizeof(int));
    }

    // Initialize matrices with random values
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i][j] = rand() % 10;
            b[i][j] = rand() % 10;
        }
    }

    // Measure start time
    clock_t start = clock();

    // Perform matrix multiplication
    multiply_matrices(a, b, c, N);

    // Measure end time
    clock_t end = clock();

    // Calculate and print the elapsed time in seconds
    double elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time taken: %f seconds\n", elapsed_time);

    // Free allocated memory
    for (int i = 0; i < N; i++) {
        free(a[i]);
        free(b[i]);
        free(c[i]);
    }
    free(a);
    free(b);
    free(c);

    return 0;
}
"""

# %%
cid = measure.PerfMeasure().create(code=code, pass_sequence=[])

runtimes = list()
for i in tqdm(range(16)):
    runtimes += [measure.PerfMeasure().measure(cid)]
r0 = statistics.fmean(runtimes)
print(r0)

# %%
cid = measure.PerfMeasure().create(code=code, pass_sequence=["-O3"])

runtimes = list()
for i in tqdm(range(16)):
    runtimes += [measure.PerfMeasure().measure(cid)]
r1 = statistics.fmean(runtimes)
print(r1)

# %%
cid = measure.PerfMeasure().create(code=code, pass_sequence=['-LICMPass', '-LoopUnrollPass', '-LoopIdiomRecognizePass', '-JumpThreadingPass', '-IndVarSimplifyPass'])

runtimes = list()
for i in tqdm(range(16)):
    runtimes += [measure.PerfMeasure().measure(cid)]
r2 = statistics.fmean(runtimes)
print(r2)

# %%
print(r0, r1, r2)

# %%
r0 / r1

# %%
r0 / r2

# %%



