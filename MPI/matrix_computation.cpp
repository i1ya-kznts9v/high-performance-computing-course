#include <iostream>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <mpi.h>

using namespace std;

void generateMatrix(vector<double>& matrix, const int n, const double min = 0.0, const double max = 1.0) {
    const double difference = max - min;
    for (int i = 0; i < n * n; i++) {
        const double random = (rand() % 101) / 100.0;
        matrix[i] = min + (random - difference / 2.0) * difference;
    }
}

void createOnesMatrix(vector<double>& matrix, const int n) {
    for (int i = 0; i < n * n; i++) {
        matrix[i] = 1.0;
    }
}

void createIdentityMatrix(vector<double>& matrix, const int n) {
    int index = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[index++] = (i == j) ? 1.0 : 0.0;
        }
    }
}

void createZerosOrOnesMatrix(vector<double>& matrix, const int n) {
    for (int i = 0; i < n * n; i++) {
        matrix[i] = rand() % 2;
    }
}

void multiplyMatrices(const vector<double>& A, const vector<double>& B, vector<double>& C, const int n, const int start_row, const int end_row) {
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0.0;
            for (int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

void multiplyMatrix(const vector<double>& A, const double c, vector<double>& B, const int n, const int start_row, const int end_row) {
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < n; j++) {
            B[i * n + j] = c * A[i * n + j];
        }
    }
}

double traceMatrix(const vector<double>& matrix, const int n, const int start_row, const int end_row) {
    double sum = 0.0;
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < n; j++) {
            sum += (i == j) ? matrix[i * n + j] : 0.0;
        }
    }
    return sum;
}

void logicalAndMatrices(const vector<double>& A, const vector<double>& B, vector<double>& C, const int n, const int start_row, const int end_row) {
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = (A[i * n + j] != 0.0 && B[i * n + j] != 0.0) ? 1.0 : 0.0;
        }
    }
}

void summarizeMatrices(const vector<double>& A, const vector<double>& B, const vector<double>& C, vector<double>& D, const int n, const int start_row, const int end_row) {
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < n; j++) {
            D[i * n + j] = A[i * n + j] + B[i * n + j] + C[i * n + j];
        }
    }
}

void fill_matrices(vector<double>& B, vector<double>& C, vector<double>& E, vector<double>& I, vector<double>& M, const int n) {
    srand(19);
    generateMatrix(B, n, 0.0, 1.0);
    generateMatrix(C, n, 0.0, 1.0);
    createOnesMatrix(E, n);
    createIdentityMatrix(I, n);
    createZerosOrOnesMatrix(M, n);
}

void bcast_matrices(vector<double>& B, vector<double>& C, vector<double>& E, vector<double>& I, vector<double>& M, const int n) {
    MPI_Bcast(B.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(C.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(E.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(I.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(M.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void compute_matrix(const vector<double>& B, const vector<double>& C, const vector<double>& E, const vector<double>& I, const vector<double>& M, vector<double>& A, const int n, const int rank, const int size) {
    int rows_per_process = n / size;
    int start_row = rank * rows_per_process;
    int end_row = (rank == size - 1) ? n : start_row + rows_per_process;

    vector<double> B2(n * n);
    vector<double> B3(n * n);
    vector<double> B3C(n * n);
    vector<double> Tr_B3C_E(n * n);
    vector<double> B_and_M(n * n);

    multiplyMatrices(B, B, B2, n, start_row, end_row);
    MPI_Allgather(MPI_IN_PLACE, rows_per_process * n, MPI_DOUBLE, B2.data(), rows_per_process * n, MPI_DOUBLE, MPI_COMM_WORLD);
    multiplyMatrices(B2, B2, B3, n, start_row, end_row);
    
    multiplyMatrices(B3, C, B3C, n, start_row, end_row);

    double Tr_B3C = traceMatrix(B3C, n, start_row, end_row);
    MPI_Allreduce(MPI_IN_PLACE, &Tr_B3C, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    multiplyMatrix(E, Tr_B3C, Tr_B3C_E, n, start_row, end_row);
    
    logicalAndMatrices(B, M, B_and_M, n, start_row, end_row);

    summarizeMatrices(Tr_B3C_E, I, B_and_M, A, n, start_row, end_row);
    MPI_Allgather(MPI_IN_PLACE, rows_per_process * n, MPI_DOUBLE, A.data(), rows_per_process * n, MPI_DOUBLE, MPI_COMM_WORLD);
}

bool equals_matrices(const vector<double>& A, const vector<double>& B, const int n, const double precision) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (abs(A[i * n + j] - B[i * n + j]) > precision) {
                return false;
            }
        }
    }
    return true;
}

void print_matrix(const vector<double>& matrix, const int n, const int view) {
    for (int i = 0; i < view; i++) {
        for (int j = 0; j < view; j++) {
            printf("%.2f ", matrix[i * n + j]);
        }
        printf("\n");
    }
}

void writeTimesToCSV(const string& filename, const vector<double>& times, const int average) {
    const bool file_existed = filesystem::exists(filename);
    ofstream file(filename, ios::app);
    if (!file.is_open()) {
        printf("Unable to write times to %s\n", filename);
    }
    if (!file_existed) {
        file << "Processes,";
        for (int j = 1; j <= average; j++) {
            file << to_string(j) << ",";
        }
        file << "Average\n";
    }
    file << int(times[0]) << ",";
    for (int i = 1; i <= average; i++) {
        file << fixed << setprecision(2) << times[i] << ",";
    }
    file << fixed << setprecision(2) << times[average + 1] << "\n";
    file.close();
}

int main(int argc, char* argv[]) {    
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        printf("Invalid arguments count\n");
        MPI_Finalize();
        return 1;
    }
    const int n = atoi(argv[1]);
    if (n < 2) {
        printf("Invalid matrices dimension\n");
        MPI_Finalize();
        return 1;
    }
    const int average = 5;

    vector<double> B(n * n);
    vector<double> C(n * n);
    vector<double> E(n * n);
    vector<double> I(n * n);
    vector<double> M(n * n);
    vector<double> A(n * n);
    vector<double> A1(n * n);
    vector<double> times(average + 2);

    if (rank == 0) fill_matrices(B, C, E, I, M, n);
    bcast_matrices(B, C, E, I, M, n);

    times[0] = size;
    double time = 0.0;
    for (int i = 0; i < average; i++) {
        if (rank == 0) size == 1 ? printf("[%d] Computing result matrix A with single process", i + 1) : printf("[%d] Computing result matrix A with %d processes", i + 1, size);

        const auto t0 = chrono::high_resolution_clock::now();
        compute_matrix(B, C, E, I, M, A, n, rank, size);
        const auto t1 = chrono::high_resolution_clock::now();
        const chrono::duration<double> duration = (t1 - t0);
        const double local_t = duration.count();
        double t = 0.0; MPI_Reduce(&local_t, &t, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        t /= size; times[i + 1] = t; time += t;
        if (rank == 0) printf(" -> %.2f (sec.)\n", t);

        if (rank == 0 && i == 0) {
            A1 = A;
            printf("[%d] Result matrix A preview:\n", i + 1);
            print_matrix(A1, n, min(5, n));
        } else if (rank == 0 && !equals_matrices(A1, A, n, 1.0E-2)) {
            printf("[%d] Result matrix A not equals to previous result matrix A\n", i + 1);
            MPI_Finalize();
            return 1;
        }
    }

    if (rank == 0) {
        printf("-------------------\n");
        printf("Teoretical statistics:\n");
        const double p = size == 1 ? 0.0 : 0.9;
        const double amdahlSpeedup = 1 / ((1 - p) + (p / size));
        printf("Amdahl speedup: %.2f\n", amdahlSpeedup);
        const double amdahlEfficency = amdahlSpeedup / size;
        printf("Amdahl efficency: %.2f\n", amdahlEfficency);
        printf("-------------------\n");
        printf("Average statistics [%d]:\n", average);
        const double tn = time / average;
        times[average + 1] = tn;
        printf("Computation time: %.2f (sec.)\n", tn);
        writeTimesToCSV("times_" + to_string(n) + ".csv", times, average);
    }
    
    MPI_Finalize();
    return 0;
}
