#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

typedef enum {
    NORMAL,
    NORMAL_BLOCKED,
    UPPER_TRIANGULAR,
    UPPER_TRIANGULAR_BLOCKED
} matrix_type_t;

typedef struct {
    int block_size;
    int blocks_in_row;
} matrix_type_info_blocked_t;

typedef union {
    matrix_type_info_blocked_t blocked_info;
} matrix_type_info_t;

typedef struct {
    matrix_type_t type;
    matrix_type_info_t minfo;
    int ncols;
    int nrows;
    void *data;
} double_matrix_t;

void matrix_fill_random(double_matrix_t m);

void matrix_mult(double_matrix_t m1, double_matrix_t m2, double_matrix_t out);

void matrix_mult_block(double_matrix_t m1, double_matrix_t m2, double_matrix_t out, int block_max_size);

static inline int matrix_contains_index(double_matrix_t m, int i, int j) {
    if (i >= m.nrows || i < 0) return 0;
    if (j >= m.ncols || j < 0) return 0;
    if (m.type == NORMAL || m.type == NORMAL_BLOCKED) return 1;
    if ((m.type == UPPER_TRIANGULAR || m.type == UPPER_TRIANGULAR_BLOCKED) && i <= j) return 1;
    return 0;
}

static inline double_matrix_t matrix_get_block(double_matrix_t m, int i, int j) {
    assert(m.type == UPPER_TRIANGULAR_BLOCKED || m.type == NORMAL_BLOCKED);
    double *plain = (double *) m.data;
    int block_size = m.minfo.blocked_info.block_size;
    int blocks_in_row = m.minfo.blocked_info.blocks_in_row;
    int block_i = i / block_size;
    int block_j = j / block_size;
    double *plain_block = plain + (block_i * blocks_in_row + block_j) * (block_size * block_size);
    return (double_matrix_t) {
        .type = NORMAL,
        .ncols = block_size,
        .nrows = block_size,
        .data = plain_block
    };
}

static inline double matrix_get(double_matrix_t m, int i, int j) {
    assert (i >= 0 && i < m.nrows && j >= 0 && j <= m.ncols);
    assert (matrix_contains_index(m, i, j));
    double *plain = (double *) m.data;
    switch (m.type)
    {
    case NORMAL:
        return plain[i * m.ncols + j];
    case UPPER_TRIANGULAR:
        return plain[j * (j + 1) / 2 + i];
    case NORMAL_BLOCKED:
    case UPPER_TRIANGULAR_BLOCKED: {
        int block_size = m.minfo.blocked_info.block_size;
        return matrix_get(
            matrix_get_block(m, i, j),
            i % block_size,
            j % block_size
        );
    }
    default:
        printf("Unknown matrices");
        assert(0);
    }
}

static double matrix_get_or_zero(double_matrix_t m, int i, int j) {
    if (!matrix_contains_index(m, i, j)) return 0;
    return matrix_get(m, i, j);
}

static inline double matrix_get_normal(double_matrix_t m, int i, int j) {
    double *plain = (double *) m.data;
    return plain[i * m.ncols + j];
}

static inline double matrix_get_upper_triangular(double_matrix_t m, int i, int j) {
    double *plain = (double *) m.data;
    return plain[j * (j + 1) / 2 + i];
}

static inline void matrix_set(double_matrix_t m, int i, int j, double value) {
    assert (i >= 0 && i < m.nrows && j >= 0 && j <= m.ncols);
    assert (matrix_contains_index(m, i, j));
    double *plain = (double *) m.data;
    switch (m.type)
    {
    case NORMAL:
        plain[i * m.ncols + j] = value;
        return;
    case UPPER_TRIANGULAR:
        plain[j * (j + 1) / 2 + i] = value;
        return;
    case NORMAL_BLOCKED:
    case UPPER_TRIANGULAR_BLOCKED: {
        int block_size = m.minfo.blocked_info.block_size;
        matrix_set(
            matrix_get_block(m, i, j),
            i % block_size,
            j % block_size,
            value
        );
        return;
    }
    default:
        printf("Unknown matrices");
        assert(0);
    }
}

static inline int matrix_print(double_matrix_t m) {
    for (int i = 0; i < m.nrows; i++) {
        for (int j = 0; j < m.ncols; j++)
            printf("%0.2f ", matrix_get_or_zero(m, i, j));
        printf("\n");
    }
    return 1;
}

static inline int matrix_equals(double_matrix_t expected, double_matrix_t actual) {
    for (int i = 0; i < actual.nrows; i++)
        for (int j = 0; j < actual.ncols; j++)
            if (((long long) matrix_get_or_zero(expected, i, j) * 1000000ll) != ((long long) matrix_get_or_zero(actual, i, j) * 1000000ll)) {
                printf("Matrices values not equals: %.6f != %.6f\n", matrix_get_or_zero(expected, i, j), matrix_get_or_zero(actual, i, j));
                return 0;
            }
    return 1;
}

static inline double_matrix_t matrix_allocate(int nrows, int ncols) {
    assert(ncols > 0 && nrows > 0);
    return (double_matrix_t) {
        .type = NORMAL,
        .ncols = ncols,
        .nrows = nrows,
        .data = calloc(1, sizeof(double) * ncols * nrows)
    };
}

static inline double_matrix_t matrix_allocate_blocked(int dims, int block_size) {
    assert(dims > 0);
    assert(dims % block_size == 0);
    return (double_matrix_t) {
        .type = NORMAL_BLOCKED,
        .minfo = (matrix_type_info_t) {
            (matrix_type_info_blocked_t) {
                .block_size = block_size,
                .blocks_in_row = dims / block_size
            }
        },
        .ncols = dims,
        .nrows = dims,
        .data = calloc(1, sizeof(double) * dims * dims)
    };
}

static inline double_matrix_t matrix_allocate_upper_triangular(int dims) {
    assert(dims > 0);
    return (double_matrix_t) {
        .type = UPPER_TRIANGULAR,
        .ncols = dims,
        .nrows = dims,
        .data = calloc(1, (sizeof(double) / 2) * (1 + dims) * dims)
    };
}

static inline double_matrix_t matrix_allocate_upper_triangular_blocked(int dims, int block_size) {
    assert(dims > 0);
    return (double_matrix_t) {
        .type = UPPER_TRIANGULAR_BLOCKED,
        .minfo = (matrix_type_info_t) { 
            (matrix_type_info_blocked_t) {
                .block_size = block_size,
                .blocks_in_row = dims / block_size
            }
        },
        .ncols = dims,
        .nrows = dims,
        .data = calloc(1, sizeof(double) * dims * dims)
    };
}

static inline void matrix_free(double_matrix_t m) {
    free(m.data);
}

static void matrix_convert(double_matrix_t m, double_matrix_t out) {
    assert(m.nrows == m.ncols);
    for (int i = 0; i < m.nrows; i++)
        for (int j = 0; j < m.nrows; j++) {
            if (matrix_contains_index(m, i, j) && matrix_contains_index(out, i, j))
                matrix_set(out, i, j, matrix_get(m, i, j));
        }
}

static double_matrix_t matrix_convert_to_normal(double_matrix_t m) {
    double_matrix_t out = matrix_allocate(m.nrows, m.ncols);
    matrix_convert(m, out);
    return out;
}

static double_matrix_t matrix_convert_to_normal_blocked(double_matrix_t m, int block_size) {
    assert(m.nrows == m.ncols);
    double_matrix_t out = matrix_allocate_blocked(m.nrows, block_size);
    matrix_convert(m, out);
    return out;
}

static double_matrix_t matrix_convert_to_upper_triangular(double_matrix_t m) {
    assert(m.nrows == m.ncols);
    double_matrix_t out = matrix_allocate_upper_triangular(m.nrows);
    matrix_convert(m, out);
    return out;
}

static double_matrix_t matrix_convert_to_upper_triangular_blocked(double_matrix_t m, int block_size) {
    assert(m.nrows == m.ncols);
    double_matrix_t out = matrix_allocate_upper_triangular_blocked(m.nrows, block_size);
    matrix_convert(m, out);
    return out;
}
