#pragma once
#include <mpi/mpi.h>
#include <cstring>
#include <iostream>

struct Matrix {
    double* matrix;
    int columns;
    int lines;
    Matrix(): columns(0), lines(0), matrix(nullptr) {} 
    Matrix(const int& lines, const int& columns) :  columns(columns), 
                                                    lines(lines), 
                                                    matrix(new double[lines * columns]()) {}

    ~Matrix() {
        delete[] matrix;
    }

    void fill();

    void operator=(const Matrix& other);

    friend std::ostream& operator<<(std::ostream &out, const Matrix& a);
};

int get_rows(const int& size, const int& rank, const int& N);
int get_start_number(const int& size, const int& rank, const int& N); 

void send_pieces_A(Matrix& matrix, const int& size, int ranks[], const int& lines, const int& columns, const int& p1, MPI_Comm Vertical_Comm);
void send_pieces_B(Matrix& B, const int& lines, const int& columns, const int& size, const int ranks[], const int& p2, MPI_Datatype COLUMN, MPI_Comm Horizontal_Comm);

double* mul_minor(Matrix& A, Matrix& B);
double* gather_full_matrix_at_null_process(const double* C, const int& n1, const int& n3, const int& p1, const int& p2, const int ranks[2], MPI_Comm Horizontal_Comm, MPI_Comm Vertical_Comm);

void broadcast_B(Matrix& B, const int& lines, const int& columns, const int& p2, MPI_Comm Vertical_Comm);
void broadcast_A(Matrix& A, const int& lines, const int& columns, const int& p1, MPI_Comm Horizontal_Comm);

void print_all(Matrix& matrix, std::string msg, const int ranks[]);