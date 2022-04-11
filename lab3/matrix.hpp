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

    void fill() {
        for(int i = 0; i < lines * columns; i++) {
            std::cin >> matrix[i];
        }
    }
    ~Matrix() {
        delete[] matrix;
    }

    Matrix& operator=(const Matrix& other) {
        columns = other.columns;
        lines = other.lines;
        matrix = new double[columns*lines];
        memcpy(matrix, other.matrix, columns*lines*sizeof(double));
    }

    friend std::ostream& operator<<(std::ostream &out, const Matrix& a){
        for (int i = 0; i < a.lines; i++) {
            for (int j = 0; j < a.columns; j++) {
                out << a.matrix[i * a.columns + j];
            }
        }
        return out;
    } 
};

int get_start_number(const int size, const int rank, const int N); 
void send_pieces_A(Matrix& matrix, const int size, const int rank, int& lines, int& columns);
void mul(double* res, const double* A, const double* B);
void print_all(Matrix& matrix, std::string msg, int rank);
int get_rows(const int size, const int rank, const int& N);
