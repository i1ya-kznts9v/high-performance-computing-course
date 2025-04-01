#include "matrix.h"
#include <omp.h>

int main(int argc, char *argv[]) {
    printf("Simple 2D-array matrix multiplication\n");

    FILE *file = fopen("simple_2d_array.csv", "w");
    if (file == NULL) return -1;
    fprintf(file, "Dim 1,Dim 2,Time (sec.)\n");

    srand(19);
    int average = 5;
    int dim_1 = 1024;
    for (int d = 9; d <= 11; d++) {
        int dim_2 = 1 << d;

        double time_sec = 0.0;
        for (int a = 1; a <= average; a++) {
            printf("[%d] A[%d][%d] * B[%d][%d]...\n", a, dim_1, dim_2, dim_2, dim_2);

            double_matrix_t A = matrix_allocate(dim_1, dim_2);
            double_matrix_t B = matrix_allocate_upper_triangular(dim_2);
            double_matrix_t result = matrix_allocate(A.nrows, B.ncols);
            matrix_fill_random(A);
            matrix_fill_random(B);

            double_matrix_t A_normal = matrix_convert_to_normal(A);
            double_matrix_t B_normal = matrix_convert_to_normal(B);
            double sec0 = omp_get_wtime();
            matrix_mult(A_normal, B_normal, result);
            double sec = omp_get_wtime() - sec0;
            time_sec += sec;

            double_matrix_t A_expected = matrix_convert_to_normal(A);
            double_matrix_t B_expected = matrix_convert_to_normal(B);
            double_matrix_t expected = matrix_allocate(A_expected.nrows, B_expected.ncols);
            matrix_mult(A_expected, B_expected, expected);
            if (!matrix_equals(expected, result)) return -1;

            matrix_free(A);
            matrix_free(A_normal);
            matrix_free(A_expected);
            matrix_free(B);
            matrix_free(B_normal);
            matrix_free(B_expected);
            matrix_free(expected);
            matrix_free(result);

            printf("[%d] Multiplication time: %.2f (sec.)\n", a, sec);
        }

        fprintf(file, "%d,%d,%.2f\n", dim_1, dim_2, time_sec / average);
        printf("Average multiplication time: %.2f (sec.)\n\n", time_sec / average);
    }
    
    fclose(file);
    return 0;
}
