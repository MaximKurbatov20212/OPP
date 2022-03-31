#include <iostream>
#include <mpi/mpi.h>
#include <string.h>
#include <vector>
#include <cmath>
#include <unistd.h>
#define t 0.00001

// Возвращает количество строк матрицы у процесса rank
int get_lrows(int N, int rank, int size){
    int lrows = N / size;
    int b = N % size;
    lrows += (rank < b); 
    return lrows;
}

// rank - ранг текущего процесса
void send_piece_of_vector_next(double* piece_of_vector, double* piece_of_vector_2, int rank, int size, int N) {
    MPI_Status status;
    int max_len_of_piece_of_vector = (N / size) + (N % size != 0);  
    if(size == 1) return;
    if(rank % 2 == 0) {
        MPI_Send(piece_of_vector, max_len_of_piece_of_vector, MPI_DOUBLE, (rank + 1) % size, 0, MPI_COMM_WORLD);
        MPI_Recv(piece_of_vector, max_len_of_piece_of_vector, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
    }
    else {
        MPI_Recv(piece_of_vector_2, max_len_of_piece_of_vector, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        MPI_Send(piece_of_vector, max_len_of_piece_of_vector, MPI_DOUBLE, (rank + 1) % size, 0, MPI_COMM_WORLD);
        memcpy(piece_of_vector, piece_of_vector_2, max_len_of_piece_of_vector * sizeof(double));
    }
}

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

// возвращает номер строки матрицы, с корого начинается кусок матрицы процесса rank
int get_number_of_first_line_in_matrix(int N, int rank, int size) {
    int sum = 0;
    for(int i = 0; i < rank; i++) {
        sum += get_lrows(N, i, size);
    }
    return sum;
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

// неполное скалярное произведение
double scalar_mul(double* vec_1, double* vec_2, int lines) {
    double res = 0.0;
    for(int i = 0; i < lines; i++) {
        res += vec_1[i] * vec_2[i]; 
    }
    return res;
}

void sub(double* R, double* A, double* B, int N) {
    for(int i = 0; i < N; i++) {
        R[i] = A[i] - B[i];
    }
}

void mul(double* res, Matrix A, double* piece_vectorX, int N, int size, int curent_rank, int len) {
    int height = A.e_row - A.s_row + 1;
    int start = get_number_of_first_line_in_matrix(N, curent_rank, size);
    
    for(int i = 0; i < height; i++) {
        res[i] += scalar_mul(A.matrix + start + i * N, piece_vectorX, len);
    }
}

double g(Matrix& A, double* piece_vectorX, double* copy_piece_vectorX, double* B, int N, int size, int rank, int number_of_lines) {
    int max_len_of_piece_of_vector = (N / size) + (N % size != 0);
    double* AX = new double[max_len_of_piece_of_vector]();
    int current_rank = rank;

    for(int i = 0; i < size; i++) {
        mul(AX, A, piece_vectorX, N, size, current_rank, get_lrows(N, current_rank, size));
        send_piece_of_vector_next(piece_vectorX, copy_piece_vectorX, rank, size, N);


        current_rank = ((current_rank - 1) + size) % size;



    }
    sub(AX, AX, B, number_of_lines);

    double piece_of_norm = scalar_mul(AX, AX, number_of_lines);
    double full_norm = 0;
    MPI_Allreduce(&piece_of_norm, &full_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    full_norm = std::sqrt(full_norm);

    double piece_of_norm_B = scalar_mul(B, B, number_of_lines);
    double full_norm_B = 0;
    MPI_Allreduce(&piece_of_norm_B, &full_norm_B, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    full_norm_B = sqrt(full_norm_B);

    delete[] AX;
    return full_norm / full_norm_B;
}

void f(double* X_n_1, Matrix& A, double* piece_vectorX, double* copy_piece_vectorX, double* B, int N, int size, int rank, int number_of_lines) {
    int max_len_of_piece_of_vector = (N / size) + (N % size != 0);
    double* AX = new double[max_len_of_piece_of_vector]();
    int current_rank = rank;

    memcpy(X_n_1, piece_vectorX, number_of_lines * sizeof(double));

    for(int i = 0; i < size; i++) {

        mul(AX, A, piece_vectorX, N, size, current_rank, get_lrows(N, current_rank, size));

        // print_vector(AX, number_of_lines, size, rank);
        send_piece_of_vector_next(piece_vectorX, copy_piece_vectorX, rank, size, N);

        // current_rank = current_rank + 1 % size;
        current_rank = ((current_rank - 1) + size) % size;
    }

    sub(AX, AX, B, number_of_lines);

    for(int i = 0; i < number_of_lines; i++) {
        AX[i] *= t;
    }

    sub(X_n_1, X_n_1, AX, number_of_lines);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = std::atoi(argv[1]);  

    Matrix A = create_matrix(N, rank, size);
    // print_(A.matrix, N, size, rank);

    int number_of_lines = get_lrows(N, rank, size);

    double* B = new double[number_of_lines]();
    for(int i = 0; i < number_of_lines; i++) B[i] = N + 1;
    
    double* copy_piece_of_vectorX = nullptr;

    int max_len_of_piece_of_vector = (N / size) + (N % size != 0);

    double* piece_of_vectorX = new double[max_len_of_piece_of_vector](); // тк будем пересылать по кругу, то все вектора должны быть одинакого размера = макс

    if(rank % 2 == 1) { // четные процессы имеют второй буфер для циклического пересыла
        copy_piece_of_vectorX = new double[max_len_of_piece_of_vector]();
    }

    // print_vector(piece_of_vectorX, get_lrows(N, rank, size), size, rank);

    double* X_n_1 = new double[max_len_of_piece_of_vector]();

    // g(A, piece_of_vectorX, copy_piece_of_vectorX, B, N, size, rank, number_of_lines);
    double start = MPI_Wtime();
    while(g(A, piece_of_vectorX, copy_piece_of_vectorX, B, N, size, rank, number_of_lines) > 0.0001) {
        f(X_n_1, A, piece_of_vectorX, copy_piece_of_vectorX, B, N, size, rank, number_of_lines);
        // print_vector(X_n_1, max_len_of_piece_of_vector, size, rank);
        memcpy(piece_of_vectorX, X_n_1, number_of_lines * sizeof(double));
        // sleep(1);
    }
    std::cout << MPI_Wtime() - start << std::endl;

    // print_vector(piece_of_vectorX, max_len_of_piece_of_vector, size, rank);
    delete[] B;
    delete[] piece_of_vectorX;
    delete[] copy_piece_of_vectorX;
    MPI_Finalize();
}