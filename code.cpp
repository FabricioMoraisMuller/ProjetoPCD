#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cblas.h>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>

// ============================================================================
// 1. CASO BASE CUSTOMIZADO COM THREADS
// ============================================================================
template <typename T>
void custom_threaded_gemm(const T *A, int lda, const T *B, int ldb, T *C, int ldc, int m, int n, int k) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; ++i) {
        for (int k_idx = 0; k_idx < k; ++k_idx) {
            T a_val = A[i * lda + k_idx];
            #pragma omp simd
            for (int j = 0; j < n; ++j) {
                C[i * ldc + j] += a_val * B[k_idx * ldb + j];
            }
        }
    }
}

inline void base_gemm(const float *A, int lda, const float *B, int ldb, float *C, int ldc, int m, int n, int k) {
    custom_threaded_gemm(A, lda, B, ldb, C, ldc, m, n, k);
}

inline void base_gemm(const double *A, int lda, const double *B, int ldb, double *C, int ldc, int m, int n, int k) {
    custom_threaded_gemm(A, lda, B, ldb, C, ldc, m, n, k);
}

// Multiplicação de referência usando OpenBLAS / CBLAS
inline void reference_gemm(const float *A, const float *B, float *C_ref, int dim) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                dim, dim, dim, 1.0f, A, dim, B, dim, 0.0f, C_ref, dim);
}

inline void reference_gemm(const double *A, const double *B, double *C_ref, int dim) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                dim, dim, dim, 1.0, A, dim, B, dim, 0.0, C_ref, dim);
}

// ============================================================================
// 2. FUNÇÃO PARA VERIFICAÇÃO DE RESULTADOS
// ============================================================================
template <typename T>
bool verificar_resultado(const T *C, const T *C_ref, int dim, T tol) {
    long total = (long)dim * dim;
    for (long i = 0; i < total; i++) {
        T diff = std::abs(C[i] - C_ref[i]);
        T rel_err = diff / (std::abs(C_ref[i]) + (T)1e-8);
        if (rel_err > tol && diff > tol) {
            return false;
        }
    }
    return true;
}

// ============================================================================
// 3. FUNÇÃO RECURSIVA DIVIDIR E CONQUISTAR
// ============================================================================
template <typename T>
void dgemm_submatrix(
    const T *A, int lda, const T *B, int ldb, T *C, int ldc,
    int m, int n, int k, int k_threshold) 
{
    if (m <= k_threshold || n <= k_threshold || k <= k_threshold) {
        base_gemm(A, lda, B, ldb, C, ldc, m, n, k);
        return;
    }

    int m2 = m / 2, n2 = n / 2, k2 = k / 2;

    const T *A11 = A, *A12 = A + k2, *A21 = A + m2 * lda, *A22 = A + m2 * lda + k2;
    const T *B11 = B, *B12 = B + n2, *B21 = B + k2 * ldb, *B22 = B + k2 * ldb + n2;
    T *C11 = C, *C12 = C + n2, *C21 = C + m2 * ldc, *C22 = C + m2 * ldc + n2;

    dgemm_submatrix(A11, lda, B11, ldb, C11, ldc, m2, n2, k2, k_threshold);
    dgemm_submatrix(A12, lda, B21, ldb, C11, ldc, m2, n2, k - k2, k_threshold);

    dgemm_submatrix(A11, lda, B12, ldb, C12, ldc, m2, n - n2, k2, k_threshold);
    dgemm_submatrix(A12, lda, B22, ldb, C12, ldc, m2, n - n2, k - k2, k_threshold);

    dgemm_submatrix(A21, lda, B11, ldb, C21, ldc, m - m2, n2, k2, k_threshold);
    dgemm_submatrix(A22, lda, B21, ldb, C21, ldc, m - m2, n2, k - k2, k_threshold);

    dgemm_submatrix(A21, lda, B12, ldb, C22, ldc, m - m2, n - n2, k2, k_threshold);
    dgemm_submatrix(A22, lda, B22, ldb, C22, ldc, m - m2, n - n2, k - k2, k_threshold);
}

template <typename T>
void gemm_divide_and_conquer(const T *A, const T *B, T *C, int dim, int k_c) {
    dgemm_submatrix(A, dim, B, dim, C, dim, dim, dim, dim, k_c);
}

// ============================================================================
// 4. BENCHMARK COMPLETO (SOLUÇÃO PRÓPRIA VS CBLAS)
// ============================================================================
template <typename T>
void executar_benchmark(std::ofstream &csv_file, const char* nome_tipo, const char* nome_funcao_cblas, int dim, const int *vetor_kc, int num_kc, T tol) {
    long total_elementos = (long)dim * dim;
    int num_threads_original = omp_get_max_threads();

    T *A = (T *)aligned_alloc(64, total_elementos * sizeof(T));
    T *B = (T *)aligned_alloc(64, total_elementos * sizeof(T));
    T *C = (T *)aligned_alloc(64, total_elementos * sizeof(T));
    T *C_ref = (T *)aligned_alloc(64, total_elementos * sizeof(T));

    srand(42);
    for (long i = 0; i < total_elementos; i++) A[i] = (T)(rand() % 100) / 10.0;
    srand(2026);
    for (long i = 0; i < total_elementos; i++) B[i] = (T)(rand() % 100) / 10.0;

    // ------------------------------------------------------------------------
    // MEDIÇÃO DO CBLAS (cblas_sgemm / cblas_dgemm)
    // ------------------------------------------------------------------------
    omp_set_num_threads(num_threads_original);
    for (long j = 0; j < total_elementos; j++) C_ref[j] = (T)0;

    double t_start_cblas = omp_get_wtime();
    reference_gemm(A, B, C_ref, dim);
    double t_cblas = omp_get_wtime() - t_start_cblas;
    double gflops_cblas = ((2.0 * (double)dim * (double)dim * (double)dim) / t_cblas) / 1e9;

    printf("\n-------------------------------------------------------------------------------------------------------------------------------\n");
    printf(" Tipo: %-6s | Matriz: %d x %d | Threads ativas (P): %d | Ref CBLAS: %s\n", nome_tipo, dim, dim, num_threads_original, nome_funcao_cblas);
    printf(" Desempenho %s: Tempo = %.4f s | GFLOPS = %.2f\n", nome_funcao_cblas, t_cblas, gflops_cblas);
    printf("-------------------------------------------------------------------------------------------------------------------------------\n");
    printf("%-8s | %-10s | %-12s | %-12s | %-12s | %-10s | %-11s | %-8s\n", 
           "k_c", "Tempo (s)", "GFLOPS", "T_1 (1 thd)", "Overhead (s)", "Speedup", "Eficiencia", "Status");
    printf("-------------------------------------------------------------------------------------------------------------------------------\n");

    for (int i = 0; i < num_kc; i++) {
        int k_c = vetor_kc[i];
        if (k_c > dim) continue;

        // 1. Tempo Sequencial Real T_1 (1 thread)
        omp_set_num_threads(1);
        for (long j = 0; j < total_elementos; j++) C[j] = (T)0;
        
        double t_start_seq = omp_get_wtime();
        gemm_divide_and_conquer(A, B, C, dim, k_c);
        double T_1 = omp_get_wtime() - t_start_seq;

        // 2. Tempo Paralelo T_P (todas as threads)
        omp_set_num_threads(num_threads_original);
        for (long j = 0; j < total_elementos; j++) C[j] = (T)0;

        double t_start_par = omp_get_wtime();
        gemm_divide_and_conquer(A, B, C, dim, k_c);
        double T_P = omp_get_wtime() - t_start_par;

        // 3. Validação do resultado contra a referência CBLAS
        bool status_ok = verificar_resultado(C, C_ref, dim, tol);

        // 4. Métricas
        double gflops = ((2.0 * (double)dim * (double)dim * (double)dim) / T_P) / 1e9;
        double speedup = T_1 / T_P;
        double eficiencia = speedup / num_threads_original;
        double overhead = (num_threads_original * T_P) - T_1;

        // Exibe na tela
        printf("%-8d | %-10.4f | %-12.2f | %-12.4f | %-12.4f | %-10.2fx | %-10.2f%% | %-8s\n",
               k_c, T_P, gflops, T_1, overhead, speedup, eficiencia * 100.0, status_ok ? "OK" : "ERRO");

        // Salva no arquivo CSV, incluindo a função de referência utilizada
        csv_file << nome_tipo << "," << dim << "," << k_c << "," << num_threads_original << ","
                 << T_P << "," << T_1 << "," << gflops << "," << overhead << ","
                 << speedup << "," << eficiencia << "," << nome_funcao_cblas << ","
                 << t_cblas << "," << gflops_cblas << "," << (status_ok ? "OK" : "ERRO") << "\n";
    }

    free(A); free(B); free(C); free(C_ref);
}

// ============================================================================
// 5. MAIN
// ============================================================================
int main(int argc, char *argv[]) {
    std::ofstream csv_file("resultados_benchmark.csv");
    if (!csv_file.is_open()) {
        std::cerr << "Erro ao criar o arquivo CSV!" << std::endl;
        return 1;
    }

    // Cabeçalho atualizado do CSV contendo a coluna 'funcao_referencia_cblas'
    csv_file << "tipo_dado,dimensao,k_c,threads,tempo_paralelo_s,tempo_seq_s,gflops,overhead_s,speedup,eficiencia,funcao_referencia_cblas,tempo_cblas_s,gflops_cblas,status\n";

    int vetor_kc[11] = {64, 96, 128, 192, 256, 384, 512, 1000, 2000, 3000, 4000};
    int tamanhos_matriz[4] = {1000, 2000, 3000, 4000};

    printf("\n=====================================================================================================================\n");
    printf("         INICIANDO BENCHMARK COMPLETO (ALL CASES, ALL K_C & CBLAS COMPARES)\n");
    printf("=====================================================================================================================\n");

    for (int t = 0; t < 4; t++) {
        int dim = tamanhos_matriz[t];

        printf("\n=====================================================================================================================\n");
        printf("                                INICIANDO TESTES PARA MATRIZ %d x %d\n", dim, dim);
        printf("=====================================================================================================================\n");

        // Passa explicitamente a string da função CBLAS de referência para float ("cblas_sgemm") e double ("cblas_dgemm")
        executar_benchmark<float>(csv_file, "float", "cblas_sgemm", dim, vetor_kc, 11, 1e-3f);
        executar_benchmark<double>(csv_file, "double", "cblas_dgemm", dim, vetor_kc, 11, 1e-7);
    }

    csv_file.close();
    printf("\n=====================================================================================================================\n");
    printf(" Resultados salvos com sucesso no arquivo CSV: resultados_benchmark.csv\n");
    printf("=====================================================================================================================\n");

    return 0;
}