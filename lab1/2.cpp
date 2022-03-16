#include <iostream>
#include <mpi/mpi.h>
#include <string.h>
#include <vector>
#include <cmath>
#include <unistd.h>
#define t 0.00001

int get_lrows(int N, int rank, int size);

void print_(double* arr, int N, int size, int rank) {
    for(int i = 0; i < size; i++){
        if(rank == i) {
            std::cout << "rank = " << rank << std::endl;
            for(int i = 0; i < get_lrows(N, rank, size); i++) {
                for(int j = 0; j < N; j++) {
                    std::cout << arr[i*N + j] << " ";
                }
                std::cout << std::endl;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

void print_vector(double* arr, int N, int size, int rank) {
    for(int i = 0; i < size; i++) {
        if(rank == i) {
            std::cout << "rank = " << rank << std::endl;
            for(int j = 0; j < N; j++) {
                std::cout << arr[j] << " ";
            }
            std::cout << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

struct Matrix {
    double* matrix;
    int s_row; // номер первой строки процесса в матрице
    int e_row; // номер последней строки процесса в матрице
    Matrix(int& N, int& s_row, int& e_row): s_row(s_row), e_row(e_row), matrix(new double[N * (e_row - s_row + 1)]) {}
};

int get_number_of_fisrt_line_in_matrix(int N, int rank, int size) {
    int sum = 0;
    for(int i = 0; i < rank; i++) {
        sum += get_lrows(N, i, size);
    }
    return sum;
}

void collect_vectors(double* X_n, double* complete_vector, int size, Matrix& A, int N, int rank) {
    int* arr = new int[size]();
    int* arr_size = new int[size]();

    for(int i = 0; i < size; i++) {
        arr_size[i] = get_lrows(N, i, size);
        
        if(i != size - 1) {
            arr[i + 1] = arr[i] + arr_size[i];
        }
    }

    MPI_Allgatherv(X_n + get_number_of_fisrt_line_in_matrix(N, rank, size) , get_lrows(N, rank, size), MPI_DOUBLE, complete_vector, arr_size, arr, MPI_DOUBLE, MPI_COMM_WORLD);
}


double norm(double* x, int N, int size) {
    double sum = 0;
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
    for(int i = 0; i < N; i++) {
        res += vec_1[i] * vec_2[i]; 
    }
    return res;
}

// Множаем матрицу на полный вектор
void mul(double* R, Matrix& A, double* B, int N) {
    double* tmp = new double[N]();

    for(int i = 0; i < N; i++) {
        tmp[i] = scalar_mul(A.matrix + i * N, B, N);
    }

    memcpy(R, tmp, N * sizeof(double));
    delete[] tmp;
}

// умножим матрицу на часть вектора
void mul(double* R, Matrix& A, double* B, int N, int rank) {
    double* tmp = new double[N]();
    int lines = (A.e_row - A.s_row + 1);
    for(int i = 0; i < lines; i++) {
        tmp[i + A.s_row] = scalar_mul(A.matrix + (i * N), B, N);
    }

    memcpy(R, tmp, N * sizeof(double));
    delete[] tmp;
}

void sub(double* R, double* A, double* B, int N) {
    for(int i = 0; i < N; i++) {
        R[i] = A[i] - B[i];
    }
}

void sub(double* R, double* A, double* B, int N, int start, int end) {
    for(int i = start; i <= end; i++) {
        R[i] = A[i] - B[i];
    }
}

void print(double* x, int N, int M){
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++) {
            int ind = index(i, j, N);
            std::cerr << x[ind] << " "; 
        }
        std::cerr << std::endl; 
    }
    std::cerr << std::endl; 
}

int get_lrows(int N, int rank, int size){
    int lrows = N / size;
    int b = N % size;
    lrows += (rank < b); 
    return lrows;
}

Matrix create_matrix(int N, int& rank, int& size) {
    int lrows = get_lrows(N, rank, size);
    int s_row = 0, e_row = 0;
    for(int i = 0; i < rank; i++) {
        s_row += get_lrows(N, i, size);
    }
    e_row = s_row + lrows - 1;
    // std::cout << "rank = " << rank << " s_row = " << s_row << " e_row = " << e_row << " lrows = " << lrows << std::endl;

    Matrix m = Matrix(N, s_row, e_row);

    // fill matrix
    for(int i = 0; i < N * lrows; i++) {
        m.matrix[i] = 1;
    }
    int a = 0;
    for(int i = s_row; i <= e_row ; i++) {
        a = i + N * (i - s_row);
        m.matrix[a] = 2;
    }
    return m;
}

// size - кол-во строк у процесса
void f(Matrix A, double* X_n, double* complete_vector, double* B, int N, int size, int rank) {
    double* X_n_1 = new double[N]();

    memcpy(X_n, complete_vector, N * sizeof(double));
    memcpy(X_n_1, complete_vector, N * sizeof(double));

    mul(X_n, A, X_n, N, rank);

    sub(X_n, X_n, B, N, A.s_row, A.e_row);

    for(int i = 0; i < N; i++) {
        X_n[i] *= t;
    }

    sub(X_n_1, X_n_1, X_n, N, A.s_row, A.e_row);


    memcpy(X_n, X_n_1, N * sizeof(double));

    collect_vectors(X_n, complete_vector, size, A, N, rank);
    delete[] X_n_1;
}

double g(Matrix A, double* X, double* B, int N, int size, int rank) {
    double* AX = new double[N]();
    double* AXB = new double[N]();

    mul(AX, A, X, N, rank); // умножаем матрицу на полный вектор

    collect_vectors(AX, AXB, size, A, N, rank); // res1 - complete vector

    sub(AXB, AXB, B, N); // от результата отнимаем B
    double n = norm(AXB, N, size); 
    double n1 = norm(B, N, size);
    
    delete[] AX;
    delete[] AXB;
    return n / n1;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = std::atoi(argv[1]);  

    Matrix A = create_matrix(N, rank, size);

    double* B = new double[N];
    for(int i = 0; i < N; i++) {
        B[i] = N + 1;
    }
    double* X = new double[N]();
    double* complete_vector = new double[N](); // собранный из кусочков вектор

    while(g(A, complete_vector, B, N, size, rank) >=  0.00001) {
        f(A, X, complete_vector, B, N, size, rank);
        // print_vector(complete_vector, N, size, rank);
        // sleep(1);
    }

    print_vector(complete_vector, N, size, rank);
    delete[] B;
    delete[] X;
    MPI_Finalize();
}