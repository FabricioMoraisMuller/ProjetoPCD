# Compile o PDF

 - Abra a pasta 'Artigo'
 - Abra o arquivo sbc-tempalte.tex
 - Aperte 'Ctrl' + 'Alt' + 'b'
 
 # Descrição das Colunas do Benchmark

* **tipo_dado**: Tipo primitivo de dado usado nos cálculos (`float` ou `double`).
* **dimensao**: Tamanho $N$ da matriz quadrada ($N \times N$).
* **k_c**: Limiar de parada da recursão do algoritmo Dividir e Conquistar.
* **threads**: Número de threads OpenMP ativas no teste ($P$).
* **tempo_paralelo_s**: Tempo de execução paralelo do algoritmo próprio ($T_P$), em segundos.
* **tempo_seq_s**: Tempo de execução sequencial do algoritmo próprio ($T_1$, 1 thread), em segundos.
* **gflops**: Taxa de bilhões de operações de ponto flutuante por segundo do algoritmo próprio.
* **overhead_s**: Tempo consumido pelo gerenciamento do paralelismo ($P \cdot T_P - T_1$), em segundos.
* **speedup**: Aceleração obtida em relação à execução sequencial ($T_1 / T_P$).
* **eficiencia**: Percentual de aproveitamento das threads ($\text{Speedup} / P$).
* **funcao_referencia_cblas**: Nome da função da biblioteca CBLAS usada como gabarito (`cblas_sgemm` ou `cblas_dgemm`).
* **tempo_cblas_s**: Tempo de execução da função CBLAS de referência, em segundos.
* **gflops_cblas**: Taxa de GFLOPS alcançada pela referência CBLAS.
* **status**: Validação do resultado calculado comparado à CBLAS (`OK` ou `ERRO`).

# Resultados
* resultados_benchmark_executado_2_threads_colab.cvs - Executado em no Colab com:
Sistema Operacional : Linux 6.6.122+
Arquitetura         : x86_64
vCPUs (Lógicas)     : 2
vCPUs (Disponíveis) : 2
Modelo do Processador: Intel(R) Xeon(R) CPU @ 2.20GHz
Memória RAM Total   : 12.67 GB