#include "matrix.h"

int min(int a, int b) {
    return (a < b) ? a : b;
}

int max(int a, int b) {
    return (a > b) ? a : b;
}

static double random_double() {
    double random = (double) rand() / RAND_MAX;
    return 2.0 * random - 1.0;
}

void matrix_fill_random(double_matrix_t m) {
    for (int i = 0; i < m.nrows; i++)
        for (int j = 0; j < m.ncols; j++)
            if (matrix_contains_index(m, i, j))
                matrix_set(m, i, j, random_double());
}

void matrix_mult(double_matrix_t m2, double_matrix_t m1, double_matrix_t out) {
    assert(out.nrows == m1.nrows && out.ncols == m2.ncols);
    assert(m1.type == m2.type && m1.type == NORMAL);
    for (int i = 0; i < out.nrows; i++) {
        for (int j = 0; j < out.ncols; j++) {
            double val = 0.0;
            for (int k = 0; k < m2.nrows; k++)
                val += matrix_get(m1, i, k) * matrix_get(m2, k, j);
            matrix_set(out, i, j, val);
        }
    }
}

void matrix_2d_mult_block(double_matrix_t m1, double_matrix_t m2, double_matrix_t out, int block_max_size) {
    assert(m1.type == UPPER_TRIANGULAR && m2.type == NORMAL);
    for (int block_start_i = 0; block_start_i < out.nrows; block_start_i += block_max_size) {
        for (int block_start_j = 0; block_start_j < out.ncols; block_start_j += block_max_size) {
            for (int block_start_k = 0; block_start_k < m2.nrows; block_start_k += block_max_size) {
                if (block_start_i > block_start_k) continue;
                for (int i = block_start_i; i < min(out.nrows, block_start_i + block_max_size); i++) {
                    for (int j = block_start_j; j < min(out.ncols, block_start_j + block_max_size); j++) {
                        double val = matrix_get(out, i, j);
                        for (int k = max(i, block_start_k); k < min(m2.nrows, block_start_k + block_max_size); k++)
                            val += matrix_get_upper_triangular(m1, i, k) * matrix_get_normal(m2, k, j);
                        matrix_set(out, i, j, val);
                    }
                }
            }
        }
    }
}

void matrix_1d_mult_block(double_matrix_t m1, double_matrix_t m2, double_matrix_t out, int block_max_size) {
    assert(m1.type == UPPER_TRIANGULAR_BLOCKED && m2.type == NORMAL_BLOCKED);
    assert(m1.minfo.blocked_info.block_size == block_max_size);
    assert(m2.minfo.blocked_info.block_size == block_max_size);
    for (int block_start_i = 0; block_start_i < out.nrows; block_start_i += block_max_size) {
        for (int block_start_j = 0; block_start_j < out.ncols; block_start_j += block_max_size) {
            for (int block_start_k = 0; block_start_k < m2.nrows; block_start_k += block_max_size) {
                if (block_start_i > block_start_k) continue;
                double_matrix_t m1_block = matrix_get_block(m1, block_start_i, block_start_k);
                double_matrix_t m2_block = matrix_get_block(m2, block_start_k, block_start_j);
                for (int i = 0; i < block_max_size; i++) {
                    for (int j = 0; j < block_max_size; j++) {
                        double val = matrix_get(out, block_start_i + i, block_start_j + j);
                        for (int k = 0; k < block_max_size; k++)
                            val += matrix_get_normal(m1_block, i, k) * matrix_get_normal(m2_block, k, j);
                        matrix_set(out, block_start_i + i, block_start_j + j, val);
                    }
                }
            }
        }
    }
}

void matrix_mult_block(double_matrix_t m2, double_matrix_t m1, double_matrix_t out, int block_max_size) {
    assert(out.nrows == m1.nrows && out.ncols == m2.ncols && m1.ncols == m2.nrows);
    assert(m1.ncols == m1.nrows && m2.ncols == m2.nrows);
    if (m1.type == UPPER_TRIANGULAR && m2.type == NORMAL) {
        matrix_2d_mult_block(m1, m2, out, block_max_size);
    } else if (m1.type == UPPER_TRIANGULAR_BLOCKED && m2.type == NORMAL_BLOCKED) {
        matrix_1d_mult_block(m1, m2, out, block_max_size);
    } else {
        printf("Unknown matrices");
        assert(0);
    }
}
