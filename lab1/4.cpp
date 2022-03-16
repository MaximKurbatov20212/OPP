#include <iostream>
#include <mpi/mpi.h>
#include <string.h>
#include <vector>
#include <cmath>
#include <unistd.h>
#define t 0.01

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
int get_number_of_fisrt_line_in_matrix(int N, int rank, int size) {
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

// N - ширина матрицы
// number_of_lines - размер вектора X
// rank - процесс, из которого вектор Х пришел
void mul(double* R, Matrix& A, double* X, int N, int number_of_lines, int rank, int size) {
    double* tmp = new double[number_of_lines]();
    memcpy(tmp, R, number_of_lines * sizeof(double));

    int start = get_number_of_fisrt_line_in_matrix(N, rank, size); // номер, с какого надо начинать читать строку в матрице
    int height_of_piece_of_matrix_A = A.e_row - A.s_row + 1;

    for(int i = 0; i < height_of_piece_of_matrix_A; i++) {
        tmp[i] += scalar_mul(A.matrix + i * N + start, X, number_of_lines);
    }

    memcpy(R, tmp, number_of_lines * sizeof(double));
    delete[] tmp;
}

void sub(double* R, double* A, double* B, int N) {
    for(int i = 0; i < N; i++) {
        R[i] = A[i] - B[i];
    }
}

// size - кол-во строк у процесса
// number_of_lines - число строк у текущего процесса
void f(Matrix A, double* X_n, double* piece_of_vector_2, int number_of_lines, double* B, int N, int size, int rank) {
    int current_rank = rank;                                // отражает ранг того процесса, чей кусочек вектора сейчас у данного процесса
    int max_len_of_piece_of_vector = (N / size) + (N % size != 0);   
    double* X_n_1 = new double[max_len_of_piece_of_vector]();
    double* D = new double[max_len_of_piece_of_vector]();
    
    memcpy(X_n_1, X_n, max_len_of_piece_of_vector * sizeof(double));
    // print_vector(X_n_1, max_len_of_piece_of_vector, size, rank);

    for(int i = 0; i < size; i++) {                                             // цикл по процессам 
        // std::cerr << "LOOP " << std::endl;
        int lines = get_lrows(N, current_rank, size);                           // количество строк у процесса
        
        // print_vector(X_n, max_len_of_piece_of_vector, size, rank);
        // std::cout << lines << std::endl;
        mul(D, A, X_n, N, lines, current_rank, size);                                 // умножили, что могли

        // print_vector(D, max_len_of_piece_of_vector, size, rank);

        send_piece_of_vector_next(X_n, piece_of_vector_2, rank, size, N);       // циклически поменяли кусочки векторов X
        current_rank = (current_rank + 1) % size;                                       // ранг процесса с которого кусочек вектора X сейчас
    }
    print_vector(D, max_len_of_piece_of_vector, size, rank);

    sub(D, D, B, number_of_lines);   
                                           // Ax - B
    for(int i = 0; i < number_of_lines; i++) D[i] *= t;                       // t(Ax - B)
    sub(X_n_1, X_n_1, D, number_of_lines);                                    // x - t(Ax - B)
    memcpy(X_n, X_n_1, max_len_of_piece_of_vector * sizeof(double));                                     // результат в X
    delete[] X_n_1;
    delete[] D;
}

double g(Matrix A, double* X_n, double* piece_of_vector_2, double* B, int N, int size, int rank, int number_of_lines) {
    int current_rank = rank;                 
    int max_len_of_piece_of_vector = (N / size) + (N % size != 0);                // отражает ранг того процесса, чей кусочек вектора сейчас у данного процесса
    double* X_n_1 = new double[max_len_of_piece_of_vector]();

    // memcpy(X_n_1, X_n, max_len_of_piece_of_vector * sizeof(double));

    for(int i = 0; i < size; i++) {                                         // цикл по процессам 
        int lines = get_lrows(N, current_rank, size);                           // количество строк у процесса
        // std::cout << lines << std::endl;
        mul(X_n_1, A, X_n, N, lines, current_rank, size);                            // умножили, что могли
        
        // print_vector(X_n_1, max_len_of_piece_of_vector, size, rank);

        send_piece_of_vector_next(X_n, piece_of_vector_2, rank, size, N);       // циклически поменяли кусочки векторов X

        current_rank = (current_rank + 1) % size;                            // ранг процесса с которого кусочек вектора X сейчас
    }

    // send_piece_of_vector_next(X_n, piece_of_vector_2, rank, size, N);       // циклически поменяли кусочки векторов X
    // print_vector(X_n_1, max_len_of_piece_of_vector, size, rank);
    sub(X_n_1, X_n_1, B, number_of_lines);   

    double piece_of_norm = scalar_mul(X_n_1, X_n_1, number_of_lines);
    double full_norm = 0;
    MPI_Allreduce(&piece_of_norm, &full_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    full_norm = std::sqrt(full_norm);
    double piece_of_norm_B = scalar_mul(B, B, number_of_lines);
    double full_norm_B = 0;
    MPI_Allreduce(&piece_of_norm_B, &full_norm_B, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    full_norm_B = sqrt(full_norm_B);
    delete[] X_n_1;
    return full_norm / full_norm_B;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = std::atoi(argv[1]);  

    Matrix A = create_matrix(N, rank, size);
    // print_(A.matrix, N, size, rank);            // +

    int number_of_lines = get_lrows(N, rank, size);

    double* B = new double[number_of_lines]();
    for(int i = 0; i < number_of_lines; i++) B[i] = N + 1;
    
    double* copy_piece_of_vectorX = nullptr;

    int max_len_of_piece_of_vector = (N / size) + (N % size != 0);                
    double* piece_of_vectorX = new double[max_len_of_piece_of_vector](); // тк будем пересылать по кругу, то все вектора должны быть одинакого размера = макс


    if(rank % 2 == 1) { // четные процессы имеют второй буфер для циклического пересыла
        copy_piece_of_vectorX = new double[max_len_of_piece_of_vector]();
    }

    while(g(A, piece_of_vectorX, copy_piece_of_vectorX, B, N, size, rank, number_of_lines) >=  0.00001) {
        // print_vector(piece_of_vectorX, max_len_of_piece_of_vector, size, rank);
        f(A, piece_of_vectorX, copy_piece_of_vectorX, number_of_lines, B, N, size, rank);
        // sleep(1);
    }

    print_vector(piece_of_vectorX, max_len_of_piece_of_vector, size, rank);
    delete[] B;
    delete[] piece_of_vectorX;
    delete[] copy_piece_of_vectorX;
    MPI_Finalize();
}