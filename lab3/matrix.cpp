#include "matrix.hpp"

int get_start_number(const int size, const int rank, const int N) {
    int start_row = 0;
    for(int i = 0; i < rank; i++) {
        start_row += get_rows(size, i, N);
    }
    return start_row;
}

int get_rows(const int size, const int rank, const int& N) {
    int lrows = N / size; // 3 / 2 = 1
    int b = N % size; // 3 % 2 = 1
    lrows += (rank < b); // 1 < 1 
    return lrows;
}

void send_pieces_A(Matrix& matrix, const int size, int ranks[], int& lines, int& columns, int p1, MPI_Comm Vertical_Comm) {
    int rows = lines / p1; // кол-во строк у процесса
    int count = rows * columns;             // кол-во элементов

    double* recv_buf = new double[count]();

    if(ranks[1] == 0) {
        MPI_Scatter(matrix.matrix, count, MPI_DOUBLE, recv_buf, count, MPI_DOUBLE, 0, Vertical_Comm);
    }

    if(ranks[0] == 0 && ranks[1] == 0) {
        matrix = Matrix(rows, columns);
        memcpy(matrix.matrix, recv_buf, count * sizeof(double));
    }    
    else if(ranks[0] != 0 && ranks[1] == 0) {
        matrix = Matrix(rows, columns);
        memcpy(matrix.matrix, recv_buf, count * sizeof(double));
    }
    else {
        matrix = Matrix(rows, columns);
    }

    delete[] recv_buf;
}

void print_all(Matrix& matrix, std::string msg, int ranks[]) {
    std::cout << "[" << ranks[0] << "," << ranks[1] << "] " << ": " << matrix << std::endl; 
    MPI_Barrier(MPI_COMM_WORLD);
}

double scalar_mul(double* A, double* B, int len) {
    double sum = 0;
    for (int i = 0; i < len; i++) {
        sum += A[i] * B[i];
    }
    return sum;
}

double* mul_minor(Matrix& A, Matrix& B) {
    double* C = new double[A.lines * B.columns]();
    for(int i = 0; i < A.lines; i++) {
        for(int j = 0; j < B.columns; j++) {
            for(int k = 0; k < A.columns; k++) {
                C[j + i * B.columns] += *(A.matrix + i * A.columns + k) * *(B.matrix + j + k * B.columns);
                // std::cout << *(A.matrix + i * A.columns + k) << " * " << *(B.matrix + j + k * B.columns) << " = " << C[j + i * B.columns] << std::endl;
            }
        }
    }
    return C;
}
