#include <iostream>
#include <string.h>
#include <cmath>
#define t 0.00001

void print_vector(double* arr, int N, int size, int rank) {
    for(int j = 0; j < N; j++) {
        std::cout << arr[j] << " ";
    }
    std::cout << std::endl;
}

double norm(double* x, int N, int size) {
    double sum = 0;

    #pragma omp parallel for reduction(+:sum)
    for(int i = 0; i < N; i++) {
        sum += x[i]*x[i];
    }
    return std::sqrt(sum);
}

inline int index(const int& i, const int& j, const int& N) {
    return i * N + j; 
}

double scalar_mul(double* vec_1, double* vec_2, int N) {
    double res = 0.0;

    // #pragma omp parallel for reduction(+:res)
    for(int i = 0; i < N; i++) {
        res += vec_1[i] * vec_2[i]; 
    }
    return res;
}

// Умножаем матрицу на полный вектор
void mul(double* R, double* A, double* B, int N) {
    double* tmp = new double[N]();

    #pragma omp parallel for schedule(static)
    for(int i = 0; i < N; i++) {
        tmp[i] = scalar_mul(A + i * N, B, N);
    }

    memcpy(R, tmp, N * sizeof(double));
    delete[] tmp;
}

void sub(double* R, double* A, double* B, int N) {
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < N; i++) {
        R[i] = A[i] - B[i];
    }
}

double* create_matrix(int N, int& rank, int& size) {
    double* matrix = new double[N*N];

    #pragma omp parallel for
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            matrix[index(i, j, N)] = 1 + (i == j);
        }
    }
    return matrix;
}

// size - кол-во строк у процесса
void f(double* A, double* X_n, double* X_n_1, double* B, int N, int size, int rank) {
    memcpy(X_n, X_n_1, N * sizeof(double));
    
    mul(X_n, A, X_n, N);

    sub(X_n, X_n, B, N);

    for(int i = 0; i < N; i++) {
        X_n[i] *= t;
    }
    
    sub(X_n_1, X_n_1, X_n, N);

    memcpy(X_n, X_n_1, N * sizeof(double));
}

double g(double* A, double* X, double* B, int N, int size, int rank) {
    double* AX = new double[N]();

    mul(AX, A, X, N); // умножаем матрицу на полный вектор
    
    sub(AX, AX, B, N); // от результата отнимаем B
    double n = norm(AX, N, size); 
    double n1 = norm(B, N, size);
    
    delete[] AX;
    return n / n1;
}

int main(int argc, char **argv) {
    int size, rank;
    rank = 0;
    size = 1;

    int N = std::atoi(argv[1]);  

    double* A = create_matrix(N, rank, size);
    
    double* B = new double[N];
    for(int i = 0; i < N; i++) {
        B[i] = N + 1;
    }

    double* X = new double[N]();
    double* X_n_1 = new double[N]();

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    // Нельзя параллелитьпараллелить
    while(g(A, X_n_1, B, N, size, rank) >=  0.00001) {
        f(A, X, X_n_1, B, N, size, rank);
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    printf("Time taken: %lf sec.\n", end.tv_sec-start.tv_sec + 0.000000001*(end.tv_nsec-start.tv_nsec));
    // print_vector(X, N, size, rank);
    delete[] B;
    delete[] X;
}