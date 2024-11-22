#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <cstdlib>
#include <omp.h>

using namespace std;

// Generate pseudo-random matrix with elements in [min, max]
void generateMatrix(vector<vector<double>>& matrix, const int n, const double min = 0.0, const double max = 1.0) {
    const double difference = max - min;
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            const double random = (rand() % 101) / 100.0;
            matrix[i][j] = min + (random - difference / 2.0) * difference;
        }
    }
}

// Create matrix with one elements
void createOnesMatrix(vector<vector<double>>& matrix, const int n) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = 1.0;
        }
    }
}

// Create matrix with one elements on diagonal
void createIdentityMatrix(vector<vector<double>>& matrix, const int n) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
}

// Create matrix with zero or one elements
void createZerosOrOnesMatrix(vector<vector<double>>& matrix, const int n) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = rand() % 2;
        }
    }
}

// Multiply matrix A on matrix B with result matrix C
void multiplyMatrices(const vector<vector<double>>& A, const vector<vector<double>>& B, vector<vector<double>>& C, const int n) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0;
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Multiply matrix A on constant c with result matrix B
void multiplyMatrix(const vector<vector<double>>& A, const double c, vector<vector<double>>& B, const int n) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            B[i][j] = c * A[i][j];
        }
    }
}

// Compute matrix trace
double traceMatrix(const vector<vector<double>>& matrix, const int n) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        sum += matrix[i][i];
    }
    return sum;
}

// Compute matrix A logical "and" matrix B with result matrix C
void logicalAndMatrices(const vector<vector<double>>& A, const vector<vector<double>>& B, vector<vector<double>>& C, const int n) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = (A[i][j] != 0.0 && B[i][j] != 0.0) ? 1.0 : 0.0;
        }
    }
}

// Compute matrix A + matrix B + matrix C with result matrix D
void summarizeMatrices(const vector<vector<double>>& A, const vector<vector<double>>& B, const vector<vector<double>>& C, vector<vector<double>>& D, const int n) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            D[i][j] = A[i][j] + B[i][j] + C[i][j];
        }
    }
}

// Fill matrices B, C, E, I, M
void fill_matrices(vector<vector<double>>& B, vector<vector<double>>& C, vector<vector<double>>& E, vector<vector<double>>& I, vector<vector<double>>& M, const int n) {
    generateMatrix(B, n, 0.0, 1.0);
    generateMatrix(C, n, 0.0, 1.0);
    createOnesMatrix(E, n);
    createIdentityMatrix(I, n);
    createZerosOrOnesMatrix(M, n);
}

// Compute result matrix A
void compute_matrix(const vector<vector<double>>& B, const vector<vector<double>>& C, const vector<vector<double>>& E, const vector<vector<double>>& I, const vector<vector<double>>& M, vector<vector<double>>& A, const int n) {
    // Create temporary matrices
    vector<vector<double>> B2(n, vector<double>(n));
    vector<vector<double>> B3(n, vector<double>(n));
    vector<vector<double>> B3C(n, vector<double>(n));
    vector<vector<double>> Tr_B3C_E(n, vector<double>(n));
    vector<vector<double>> B_and_M(n, vector<double>(n));

    // Compute B^3
    multiplyMatrices(B, B, B2, n);
    multiplyMatrices(B2, B, B3, n);
    // Compute B^3 * C
    multiplyMatrices(B3, C, B3C, n);
    // Compute Tr(B^3 * C)
    double Tr_B3C = traceMatrix(B3C, n);
    // Compute Tr(B^3 * C) * E
    multiplyMatrix(E, Tr_B3C, Tr_B3C_E, n);
    // Compute B && M
    logicalAndMatrices(B, M, B_and_M, n);
    // Compute Tr(B^3 * C) * E + I + B && M
    summarizeMatrices(Tr_B3C_E, I, B_and_M, A, n);
}

// Check equivalence of matrix A and matrix B with precision
bool equals_matrices(const vector<vector<double>>& A, const vector<vector<double>>& B, const int n, const double precision) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (abs(A[i][j] - B[i][j]) > precision) {
                return false;
            }
        }
    }
    return true;
}

// Print matrix
void print_matrix(const vector<vector<double>>& matrix, const int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f ", matrix[i][j]);
        }
        printf("\n");
    }
}

// Write times to CSV
void writeTimesToCSV(const string& filename, const vector<vector<double>>& times, int pthreads, int average) {
    ofstream file(filename);
    if (!file.is_open()) {
        printf("Unable to write times to %s\n", filename);
    }
    // CSV header
    file << "Threads,";
    for (int j = 1; j <= average; j++) {
        file << to_string(j) << ",";
    }
    file << "Average\n";
    // Write times
    for (int i = 0; i < pthreads; i++) {
        file << int(times[i][0]) << ",";
        for (int j = 1; j <= average; j++) {
            file << fixed << setprecision(2) << times[i][j] << ",";
        }
        file << fixed << setprecision(2) << times[i][average + 1] << "\n";
    }
    file.close();
}

// Write statistics to CSV
void writeStatisticsToCSV(const string& filename, const vector<tuple<int, double, double, double, double>>& statistics) {
    ofstream file(filename);
    if (!file.is_open()) {
        printf("Unable to write statistics to %s\n", filename);
    }
    // CSV header
    file << "Threads,Amdahl speedup,Speedup,Amdahl efficency,Efficency\n";
    // Write statistics
    for (const auto& [nthreads, amdahlSpeedup, speedup, amdahlEfficency, efficency] : statistics) {
        file << fixed << setprecision(2) << nthreads << "," << amdahlSpeedup << "," << speedup << "," << amdahlEfficency << "," << efficency << "\n";
    }
    file.close();
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Invalid arguments count\n");
        return 1;
    }
    int n = atoi(argv[1]); // Setting user's matrices dimension
    if (n < 2) {
        printf("Invalid matrices dimension\n");
        return 1;
    }
    const int pthreads = 8;
    const int average = 5;

    // Create matrices and variables
    vector<vector<double>> B(n, vector<double>(n));
    vector<vector<double>> C(n, vector<double>(n));
    vector<vector<double>> E(n, vector<double>(n));
    vector<vector<double>> I(n, vector<double>(n));
    vector<vector<double>> M(n, vector<double>(n));
    vector<vector<double>> A(n, vector<double>(n));
    vector<vector<double>> A1(n, vector<double>(n));
    vector<vector<double>> times(pthreads, vector<double>(average + 2));
    vector<tuple<int, double, double, double, double>> statistics;
    double t1 = 0.0;

    // Fill matrices
    srand(19); // Setting the pseudo-random generation seed for reproducibility of results
    fill_matrices(B, C, E, I, M, n);

    // Compute result matrices A with power of two number of threads
    omp_set_dynamic(0); // Disable dynamic teams to force number of threads setting
    for (int i = 0; i < pthreads; i++) {
        const int nthreads = 1 << i; // Setting power of two number of threads
        times[i][0] = nthreads;
        double time = 0.0;
        for (int j = 0; j < average; j++) {
            omp_set_num_threads(nthreads); // Setting number of threads for all subsequent parallel regions
            nthreads == 1 ? printf("[%d] Computing result matrix A with single thread", j + 1) : printf("[%d] Computing result matrix A with %d threads", j + 1, nthreads);

            const double t0 = omp_get_wtime();
            compute_matrix(B, C, E, I, M, A, n);
            const double t = omp_get_wtime() - t0;
            times[i][j + 1] = t;
            time += t;
            printf(" -> %.2f (sec.)\n", t);

            if (nthreads == 1 && j == 0) {
                A1 = A; // Save result matrix A with single thread
                printf("Result matrix A preview:\n");
                print_matrix(A1, min(5, n)); // Preview result matrix A with single thread
            } else if (!equals_matrices(A1, A, n, 1.0E-2)) { // Compare result matrix A with single thread and result matrix A with current number of threads
                printf("Result matrix A with single thread not equals to result matrix A with %d threads\n", nthreads);
                return 1;
            }
        }

        printf("-------------------\n");
        printf("Teoretical statistics:\n");
        const double p = nthreads == 1 ? 0.0 : 0.9;
        const double amdahlSpeedup = 1 / ((1 - p) + (p / nthreads));
        printf("Amdahl speedup: %.2f\n", amdahlSpeedup);
        const double amdahlEfficency = amdahlSpeedup / nthreads;
        printf("Amdahl efficency: %.2f\n", amdahlEfficency);
        printf("-------------------\n");
        printf("Average statistics [%d]:\n", average);
        const double tn = time / average;
        times[i][average + 1] = tn;
        if (nthreads == 1) t1 = tn;
        printf("Computation time: %.2f (sec.)\n", tn);
        const double speedup = t1 / tn;
        printf("Speedup: %.2f\n", speedup);
        const double efficency = speedup / nthreads;
        printf("Efficency: %.2f\n\n", efficency);
        statistics.emplace_back(nthreads, amdahlSpeedup, speedup, amdahlEfficency, efficency);
    }

    writeTimesToCSV("times_" + to_string(n) + ".csv", times, pthreads, average);
    writeStatisticsToCSV("statistics_" + to_string(n) + ".csv", statistics);
    return 0;
}
