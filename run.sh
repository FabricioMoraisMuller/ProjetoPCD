#!/bin/bash
set -e

SRC="code.cpp"
OUTPUT="gemm_benchmark"

echo "Compilando ${SRC} com g++ (OpenBLAS + OpenMP)..."
g++ -O3 -march=native -fopenmp "${SRC}" -o "${OUTPUT}" -lopenblas

echo "Compilação concluída! Executando com parâmetros: $@"
echo "--------------------------------------------"
./"${OUTPUT}" "$@"