#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdint.h>
#include <time.h>
#include <windows.h>
#include <omp.h>

#define threads 8
int RunLoop = 1;

double GetTimeSeconds(void) {
    static LARGE_INTEGER frequency;
    LARGE_INTEGER counter;

    if (frequency.QuadPart == 0) {
        QueryPerformanceFrequency(&frequency);
    }

    QueryPerformanceCounter(&counter);

    return (double)counter.QuadPart /
           (double)frequency.QuadPart;
}

// Allocate a zero-initialized square matrix safely.
int* AllocateMatrix(int n) {
    if (n <= 0 ||
        (size_t)n > SIZE_MAX / sizeof(int) / (size_t)n) {
        return NULL;
    }

    return calloc((size_t)n * n, sizeof(int));
}

int NextPowerTwo(int n) {
    int siz = 1;

    if (n <= 0) return 0;

    while (siz < n) {
        if (siz > INT_MAX / 2) return 0;
        siz *= 2;
    }

    return siz;
}

void PrintMatrix(const int* arr, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", arr[n * i + j]);
        }
        printf("\n");
    }
}

void SeedMatrix(int* arr, int n) {
    for (int i = 0; i < n * n; i++) {
        arr[i] = (rand() % 2001) - 1000;
    }
}

// Returns a NEW matrix. The caller retains ownership of arr.
int* ResizeMatrix(const int* arr, int n, int siz) {
    int* arr2 = AllocateMatrix(siz);
    if (!arr2) return NULL;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            arr2[i * siz + j] = arr[i * n + j];
        }
    }

    return arr2;
}

int* MatrixAdd(const int* arr1, const int* arr2, int n) {
    int* arr3 = AllocateMatrix(n);
    if (!arr3) return NULL;

    for (int i = 0; i < n * n; i++) {
        arr3[i] = arr1[i] + arr2[i];
    }

    return arr3;
}

int* MatrixSub(const int* arr1, const int* arr2, int n) {
    int* arr3 = AllocateMatrix(n);
    if (!arr3) return NULL;

    for (int i = 0; i < n * n; i++) {
        arr3[i] = arr1[i] - arr2[i];
    }

    return arr3;
}

int* GetQuadrant(const int* arr, int n, int row, int col) {
    int siz = n / 2;
    int* arr2 = AllocateMatrix(siz);
    if (!arr2) return NULL;

    for (int i = 0; i < siz; i++) {
        for (int j = 0; j < siz; j++) {
            arr2[i * siz + j] =
                arr[(i + row * siz) * n + (j + col * siz)];
        }
    }

    return arr2;
}

int* MergeQuadrants(
    const int* c11,
    const int* c12,
    const int* c21,
    const int* c22,
    int n
) {
    int siz = n / 2;
    int* arr = AllocateMatrix(n);
    if (!arr) return NULL;

    for (int i = 0; i < siz; i++) {
        for (int j = 0; j < siz; j++) {
            arr[i * n + j] = c11[i * siz + j];
            arr[i * n + j + siz] = c12[i * siz + j];
            arr[(i + siz) * n + j] = c21[i * siz + j];
            arr[(i + siz) * n + j + siz] =
                c22[i * siz + j];
        }
    }

    return arr;
}

int* StrassenMM(const int* arr1, const int* arr2, int n) {
    if (n == 1) {
        int* result = AllocateMatrix(1);
        if (result) {
            result[0] = arr1[0] * arr2[0];
        }
        return result;
    }

    int* a11 = NULL; int* a12 = NULL;
    int* a21 = NULL; int* a22 = NULL;
    int* b11 = NULL; int* b12 = NULL;
    int* b21 = NULL; int* b22 = NULL;

    int* m1 = NULL; int* m2 = NULL;
    int* m3 = NULL; int* m4 = NULL;
    int* m5 = NULL; int* m6 = NULL;
    int* m7 = NULL;

    int* c11 = NULL; int* c12 = NULL;
    int* c21 = NULL; int* c22 = NULL;

    int* x = NULL;
    int* y = NULL;
    int* result = NULL;

    int siz = n / 2;

    a11 = GetQuadrant(arr1, n, 0, 0);
    a12 = GetQuadrant(arr1, n, 0, 1);
    a21 = GetQuadrant(arr1, n, 1, 0);
    a22 = GetQuadrant(arr1, n, 1, 1);

    b11 = GetQuadrant(arr2, n, 0, 0);
    b12 = GetQuadrant(arr2, n, 0, 1);
    b21 = GetQuadrant(arr2, n, 1, 0);
    b22 = GetQuadrant(arr2, n, 1, 1);

    if (!a11 || !a12 || !a21 || !a22 ||
        !b11 || !b12 || !b21 || !b22) {
        goto cleanup;
    }


    if (n >= 128 && omp_in_parallel()) {

        #pragma omp task shared(m1) firstprivate(a11, a22, b11, b22, siz)
        {
            int* x = MatrixAdd(a11, a22, siz);
            int* y = MatrixAdd(b11, b22, siz);

            if (x && y)
                m1 = StrassenMM(x, y, siz);

            free(x);
            free(y);
        }

        #pragma omp task shared(m2) firstprivate(a21, a22, b11, siz)
        {
            int* x = MatrixAdd(a21, a22, siz);

            if (x)
                m2 = StrassenMM(x, b11, siz);

            free(x);
        }

        #pragma omp task shared(m3) firstprivate(a11, b12, b22, siz)
        {
            int* y = MatrixSub(b12, b22, siz);

            if (y)
                m3 = StrassenMM(a11, y, siz);

            free(y);
        }

        #pragma omp task shared(m4) firstprivate(a22, b21, b11, siz)
        {
            int* y = MatrixSub(b21, b11, siz);

            if (y)
                m4 = StrassenMM(a22, y, siz);

            free(y);
        }

        #pragma omp task shared(m5) firstprivate(a11, a12, b22, siz)
        {
            int* x = MatrixAdd(a11, a12, siz);

            if (x)
                m5 = StrassenMM(x, b22, siz);

            free(x);
        }

        #pragma omp task shared(m6) firstprivate(a21, a11, b11, b12, siz)
        {
            int* x = MatrixSub(a21, a11, siz);
            int* y = MatrixAdd(b11, b12, siz);

            if (x && y)
                m6 = StrassenMM(x, y, siz);

            free(x);
            free(y);
        }

        #pragma omp task shared(m7) firstprivate(a12, a22, b21, b22, siz)
        {
            int* x = MatrixSub(a12, a22, siz);
            int* y = MatrixAdd(b21, b22, siz);

            if (x && y)
                m7 = StrassenMM(x, y, siz);

            free(x);
            free(y);
        }

        #pragma omp taskwait

    } else {
        // Sequential execution: no new OpenMP tasks.

        x = MatrixAdd(a11, a22, siz);
        y = MatrixAdd(b11, b22, siz);
        if (x && y)
            m1 = StrassenMM(x, y, siz);
        free(x); x = NULL;
        free(y); y = NULL;

        x = MatrixAdd(a21, a22, siz);
        if (x)
            m2 = StrassenMM(x, b11, siz);
        free(x); x = NULL;

        y = MatrixSub(b12, b22, siz);
        if (y)
            m3 = StrassenMM(a11, y, siz);
        free(y); y = NULL;

        y = MatrixSub(b21, b11, siz);
        if (y)
            m4 = StrassenMM(a22, y, siz);
        free(y); y = NULL;

        x = MatrixAdd(a11, a12, siz);
        if (x)
            m5 = StrassenMM(x, b22, siz);
        free(x); x = NULL;

        x = MatrixSub(a21, a11, siz);
        y = MatrixAdd(b11, b12, siz);
        if (x && y)
            m6 = StrassenMM(x, y, siz);
        free(x); x = NULL;
        free(y); y = NULL;

        x = MatrixSub(a12, a22, siz);
        y = MatrixAdd(b21, b22, siz);
        if (x && y)
            m7 = StrassenMM(x, y, siz);
        free(x); x = NULL;
        free(y); y = NULL;
    }

    if (!m1 || !m2 || !m3 || !m4 ||
        !m5 || !m6 || !m7) {
        goto cleanup;
    }


    if (!m1 || !m2 || !m3 || !m4 ||
    !m5 || !m6 || !m7) {
    goto cleanup;
    }

    // C11 = M1 + M4 - M5 + M7
    x = MatrixAdd(m1, m4, siz);
    if (!x) goto cleanup;
    y = MatrixSub(x, m5, siz);
    free(x); x = NULL;
    if (!y) goto cleanup;
    c11 = MatrixAdd(y, m7, siz);
    free(y); y = NULL;
    if (!c11) goto cleanup;

    // C12 = M3 + M5
    c12 = MatrixAdd(m3, m5, siz);
    if (!c12) goto cleanup;

    // C21 = M2 + M4
    c21 = MatrixAdd(m2, m4, siz);
    if (!c21) goto cleanup;

    // C22 = M1 - M2 + M3 + M6
    x = MatrixAdd(m1, m3, siz);
    if (!x) goto cleanup;
    y = MatrixSub(x, m2, siz);
    free(x); x = NULL;
    if (!y) goto cleanup;
    c22 = MatrixAdd(y, m6, siz);
    free(y); y = NULL;
    if (!c22) goto cleanup;

    result = MergeQuadrants(c11, c12, c21, c22, n);

cleanup:
    // Every temporary allocation is released here.
    free(a11); free(a12); free(a21); free(a22);
    free(b11); free(b12); free(b21); free(b22);

    free(m1); free(m2); free(m3); free(m4);
    free(m5); free(m6); free(m7);

    free(c11); free(c12); free(c21); free(c22);
    free(x); free(y);

    return result;
}

int main(void) {
    srand((unsigned int)time(NULL));

    omp_set_dynamic(0);
    omp_set_num_threads(threads);

    while (RunLoop) {
        int n;

        printf("Choose the matrix size (0 to exit): ");

        if (scanf("%d", &n) != 1) {
            fprintf(stderr, "Invalid input.\n");
            break;
        }

        if (n == 0) break;

        if (n < 0) {
            fprintf(stderr, "Matrix size must be positive.\n");
            continue;
        }

        int siz = NextPowerTwo(n);
        if (siz == 0) {
            fprintf(stderr, "Matrix size is too large.\n");
            continue;
        }

        int* original1 = AllocateMatrix(n);
        int* original2 = AllocateMatrix(n);

        if (!original1 || !original2) {
            fprintf(stderr, "Failed to allocate input matrices.\n");
            free(original1);
            free(original2);
            continue;
        }

        SeedMatrix(original1, n);
        SeedMatrix(original2, n);

        int* arr1 = ResizeMatrix(original1, n, siz);
        int* arr2 = ResizeMatrix(original2, n, siz);

        free(original1);
        free(original2);

        if (!arr1 || !arr2) {
            fprintf(stderr, "Failed to resize matrices.\n");
            free(arr1);
            free(arr2);
            continue;
        }

        // 1. Sequential baseline
        double start_seq = omp_get_wtime();
        int* seq_result = StrassenMM(arr1, arr2, siz);
        double end_seq = omp_get_wtime();

        if (!seq_result) {
            fprintf(stderr, "Sequential multiplication failed.\n");
            free(arr1);
            free(arr2);
            continue;
        }

        double sequential_time = end_seq - start_seq;

        // 2. Parallel multiplication
        int* par_result = NULL;

        double start_par = omp_get_wtime();

        #pragma omp parallel num_threads(threads)
        {
            #pragma omp single
            {
                par_result = StrassenMM(arr1, arr2, siz);
            }
        }

        double end_par = omp_get_wtime();
        double parallel_time = end_par - start_par;

        if (!par_result) {
            fprintf(stderr, "Parallel multiplication failed.\n");
            free(seq_result);
            free(arr1);
            free(arr2);
            continue;
        }

        // 3. Validate that the results match
        int correct = 1;

        for (int i = 0; i < siz * siz; i++) {
            if (seq_result[i] != par_result[i]) {
                correct = 0;
                break;
            }
        }

        // 4. Calculate metrics
        double overhead = parallel_time - sequential_time;
        double speedup = 0.0;
        double efficiency = 0.0;

        if (parallel_time > 0.0) {
            speedup = sequential_time / parallel_time;
            efficiency = speedup / threads;
        }

        printf("\n--- Metricas ---\n");
        printf("Matriz original: %d x %d\n", n, n);
        printf("Matriz utilizada: %d x %d\n", siz, siz);
        printf("Threads solicitadas: %d\n", threads);
        printf("Resultado correto: %s\n",
               correct ? "Sim" : "Nao");

        printf("Tempo sequencial: %.6f segundos\n",
               sequential_time);
        printf("Tempo paralelo: %.6f segundos\n",
               parallel_time);

        printf("Overhead paralelo (Tp - Ts): %.6f segundos\n",
               overhead);

        printf("Speedup: %.3fx\n", speedup);
        printf("Eficiencia: %.2f%%\n", efficiency * 100.0);

        free(seq_result);
        free(par_result);
        free(arr1);
        free(arr2);
    }

    return 0;
}