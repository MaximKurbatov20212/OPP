#include "matrix.hpp"
#include <unistd.h>

void send_pieces_A(Matrix& matrix, const int& size, int ranks[], const int& lines, const int& columns, const int& p1, MPI_Comm Vertical_Comm) {
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

void print_all(Matrix& matrix, std::string msg, const int ranks[]) {
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
            // std::cout << "--------------------------------------" << std::endl;
            for(int k = 0; k < A.columns; k++) {
                C[j + i * B.columns] += *(A.matrix + i * A.columns + k) * *(B.matrix + j + k * B.columns);
                // std::cout << *(A.matrix + i * A.columns + k) << " * " << *(B.matrix + j + k * B.columns) << " = " << C[j + i * B.columns] << std::endl;
            }
        }
    }
    return C;
}

void send_pieces_B(Matrix& B, const int& lines, const int& columns, const int& size, const int ranks[], const int& p2, MPI_Datatype COLUMN, MPI_Comm Horizontal_Comm) {
    int count = columns / p2;
    double* my_column = new double[lines * count];

    if(ranks[0] == 0 && ranks[1] == 0) {
        // Send the column
        MPI_Scatter(B.matrix, 1, COLUMN, my_column, lines * count, MPI_DOUBLE, 0, Horizontal_Comm);
        B = Matrix(lines, count);
        memcpy(B.matrix, my_column, lines*count* sizeof(double));
    }

    else if(ranks[0] == 0 && ranks[1] != 0) {
        // Receive the column
        MPI_Scatter(NULL, 1, COLUMN, my_column, lines * count, MPI_DOUBLE, 0, Horizontal_Comm);
        B = Matrix(lines, count);
        memcpy(B.matrix, my_column, lines*count* sizeof(double));
    }
    else {
        B = Matrix(lines, count);
    }

    delete[] my_column;
}

void broadcast_A(Matrix& A, const int& lines, const int& columns, const int& p1, MPI_Comm Horizontal_Comm) {
    int count = lines / p1 * columns;           
    MPI_Bcast(A.matrix, count, MPI_DOUBLE, 0, Horizontal_Comm);
}

void broadcast_B(Matrix& B, const int& lines, const int& columns, const int& p2, MPI_Comm Vertical_Comm) {
    int count = columns / p2 * lines;
    MPI_Bcast(B.matrix, count, MPI_DOUBLE, 0, Vertical_Comm);
}

double* gather_full_matrix_at_null_process(const double* C, const int& n1, const int& n3, const int& p1, const int& p2, const int ranks[2], MPI_Comm Horizontal_Comm, MPI_Comm Vertical_Comm) {
    double* vertical_result = new double[n1 / p1 * n3]();

    for(int i = 0; i < n1 / p1; i++) { // цикл по строчкам
        MPI_Gather(C + i * (n3 / p2), n3 / p2, MPI_DOUBLE, vertical_result + i * n3, n3 / p2, MPI_DOUBLE, 0, Horizontal_Comm);
    }

    // sleep(ranks[0]);
    // for(int i = 0; i < n1 / p1 * n3; i++) {
    //     std::cout << vertical_result[i] << " ";
    // }
    // MPI_Barrier(MPI_COMM_WORLD);

    double* horizontal_result = new double[n1 * n3]();

    MPI_Gather(vertical_result, n1 / p1 * n3 , MPI_DOUBLE, horizontal_result, n1 / p1 * n3 , MPI_DOUBLE, 0, Vertical_Comm);

    delete[] vertical_result;
    return horizontal_result;
}


void Matrix::operator=(const Matrix& other) {
    columns = other.columns;
    lines = other.lines;
    delete[] matrix;
    matrix = new double[columns*lines];
    memcpy(matrix, other.matrix, columns*lines*sizeof(double));
}

std::ostream& operator<<(std::ostream &out, const Matrix& a){
    for (int i = 0; i < a.lines; i++) {
        for (int j = 0; j < a.columns; j++) {
            out << a.matrix[i * a.columns + j] << " ";
        }
    }
    return out;
} 

void Matrix::fill() {
    for(int i = 0; i < lines * columns; i++) {
        std::cin >> matrix[i];
    }
}