#include "matrix.h"
#include <omp.h>

int main(int argc, char *argv[]) {
    printf("Blocked 1D-array matrix multiplication\n");

    FILE *file = fopen("blocked_1d_array.csv", "w");
    if (file == NULL) return -1;
    fprintf(file, "Dim 1,Dim 2,Time (sec.),Block\n");

    srand(19);
    int average = 5;
    int dims_1[] = {1024, 2880};
    int dims_2[] = {1024, 2048, 2880};
    for (int d = 0; d <= 2; d++) {
        int dim_1 = dims_1[0]; if (d == 2) dim_1 = dims_1[1];
        int dim_2 = dims_2[d];

        int blocks[] = {1, 2, 4, 8, 16, 32, 64, 144, 288, 576, 1440};
        for (int b = 0; b <= 10; b++) {
            int block = 1 << b; if (d == 2) block = blocks[b];

            double time_sec = 0.0;
            for (int a = 1; a <= average; a++) {
                printf("[%d] A[%d/%d][%d/%d] * B[%d/%d][%d/%d]...\n",
                    a, dim_1, block, dim_2, block, dim_2, block, dim_2, block);
    
                double_matrix_t A = matrix_allocate(dim_1, dim_2);
                double_matrix_t B = matrix_allocate_upper_triangular(dim_2);
                double_matrix_t result = matrix_allocate(A.nrows, B.ncols);
                matrix_fill_random(A);
                matrix_fill_random(B);
    
                double_matrix_t A_blocked = matrix_convert_to_normal_blocked(A, block);
                double_matrix_t B_blocked = matrix_convert_to_upper_triangular_blocked(B, block);
                double sec0 = omp_get_wtime();
                matrix_mult_block(A_blocked, B_blocked, result, block);
                double sec = omp_get_wtime() - sec0;
                time_sec += sec;
    
                double_matrix_t A_expected = matrix_convert_to_normal(A);
                double_matrix_t B_expected = matrix_convert_to_normal(B);
                double_matrix_t expected = matrix_allocate(A_expected.nrows, B_expected.ncols);
                matrix_mult(A_expected, B_expected, expected);
                if (!matrix_equals(expected, result)) return -1;

                matrix_free(A);
                matrix_free(A_blocked);
                matrix_free(A_expected);
                matrix_free(B);
                matrix_free(B_blocked);
                matrix_free(B_expected);
                matrix_free(expected);
                matrix_free(result);
    
                printf("[%d] Multiplication time: %.2f (sec.)\n", a, sec);
            }

            fprintf(file, "%d,%d,%.2f,%d\n", dim_1, dim_2, time_sec / average, block);
            printf("Average multiplication time: %.2f (sec.)\n\n", time_sec / average);
        }
    }
    
    fclose(file);
    return 0;
}
