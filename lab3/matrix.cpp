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

void send_pieces_A(Matrix& matrix, const int size, const int rank, int& lines, int& columns) {
    int rows = get_rows(size, rank, lines); // кол-во строк у процесса
    int count = rows * columns;             // кол-во элементов
    double* recv_buf = new double[count]();

    MPI_Scatter(matrix.matrix, count, MPI_DOUBLE, recv_buf, count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if(rank != 0) {
        matrix = Matrix(rows, columns);
        memcpy(matrix.matrix, recv_buf, count * sizeof(double));
    }
}

void send_pieces_B(Matrix& matrix, const int size, const int rank, int& lines, int& columns, int p2) {
    int piece_columns = get_rows(size, rank, columns); // кол-во столбцов у процесса
    int count = piece_columns * lines;             // кол-во элементов
    MPI_Datatype COLUMNS_B;
    MPI_Type_vector(lines * (columns / p2), count, count , MPI_DOUBLE, &COLUMNS_B); 
    double* recv_buf = new double[count]();

    MPI_Scatter(matrix.matrix, count, COLUMNS_B, recv_buf, count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if(rank != 0) {
        matrix = Matrix(rows, columns);
        memcpy(matrix.matrix, recv_buf, count * sizeof(double));
    }
}

void print_all(Matrix& matrix, std::string msg, int rank) {
    std::cout << rank << ": " << matrix << std::endl; 
    MPI_Barrier;
}

// void mul_minors(double* R, Matrix& A, B_TYPE* B) {
//     for(int i = 0; i < A.lines; i++) {
//         for(int j = 0; j < B.columns; j++) {
//             R += scalar_mul(A.matrix + i * A.columns, B, );
//         }
//     }
// 
// }
