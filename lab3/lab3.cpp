#include <iostream>
#include <mpi/mpi.h>
#include <unistd.h>
#include "matrix.hpp"

MPI_Comm create_grid_of_process(const int& size, const int& rank, int& p1, int& p2) {
    int arr_dims[2]{p1, p2};
    int arr_period[2]{1,1};
    MPI_Comm My_Comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, arr_dims, arr_period, 0, &My_Comm);

    int my_ranks[2]{0,0};
    MPI_Cart_coords(My_Comm, rank, 2, my_ranks);
    //print_(my_ranks);

    MPI_Comm new_comm;
    const int remain_dims[2]{true, false};
    MPI_Cart_sub(My_Comm, remain_dims, &new_comm);
    return My_Comm;
}

void get_new_datatype(int lines, int columns, int p2, MPI_Datatype* B) {
    MPI_Datatype COLUMN_NOT_RESIZED;
    MPI_Type_vector(lines, columns / p2, columns , MPI_DOUBLE, &COLUMN_NOT_RESIZED);
    MPI_Type_commit(&COLUMN_NOT_RESIZED);
    MPI_Type_create_resized(COLUMN_NOT_RESIZED, 0, (columns / p2) * sizeof(double), B);
    MPI_Type_commit(B);
}

void send_pieces_B(Matrix& B, int lines, int columns, int size, int ranks[], int p2, MPI_Datatype COLUMN, MPI_Comm Horizontal_Comm) {
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
    // printf("MPI process %d received column made of cells %f %f %f, %f, %f, %f, %f, %f\n", rank, my_column[0], my_column[1], my_column[2], my_column[3], my_column[4], my_column[5], my_column[6], my_column[7]);
    delete[] my_column;
}

MPI_Comm create_horizontal_comm(MPI_Comm Old_Comm) {
    MPI_Comm New_Comm;
    int remain_dims[2]{false, true};
    MPI_Cart_sub(Old_Comm, remain_dims, &New_Comm);
    return New_Comm;
}

MPI_Comm create_vertical_comm(MPI_Comm Old_Comm) {
    MPI_Comm New_Comm;
    int remain_dims[2]{true, false};
    MPI_Cart_sub(Old_Comm, remain_dims, &New_Comm);
    return New_Comm;
}


void broadcast_A(Matrix& A, int lines, int columns, int p1, MPI_Comm Horizontal_Comm) {
    int count = lines / p1 * columns;           
    MPI_Bcast(A.matrix, count, MPI_DOUBLE, 0, Horizontal_Comm);
}

void broadcast_B(Matrix& B, int lines, int columns, int p2, MPI_Comm Vertical_Comm) {
    int count = columns / p2 * lines;
    MPI_Bcast(B.matrix, count, MPI_DOUBLE, 0, Vertical_Comm);
}

double* gather_full_matrix_at_null_process(double* C, int n1, int n3, int p1, int p2, int ranks[2], MPI_Comm Horizontal_Comm, MPI_Comm Vertical_Comm) {
    double* vertical_result = new double[n1 / p1 * n3]();
    // std::cout << "n1 / p2 = " << n1 / p2 << std::endl;
    for(int i = 0; i < n1 / p2; i++) {
        MPI_Gather(C + i * (n3 / p2), n3 / p2, MPI_DOUBLE, vertical_result + i * n3,  n3 / p2, MPI_DOUBLE, 0, Horizontal_Comm);
    }

        // MPI_Gather(C, n1 / p1 * n3 / p2, MPI_DOUBLE, R1, n1 / p1 * n3 / p2, MPI_DOUBLE, 0, Horizontal_Comm);
    // std::cout << "[" << ranks[0] << "," << ranks[1] << "]" << std::endl; 
    // for(int i = 0; i < n1 / p1 * n3; i++) {
    //     std::cout << vertical_result[i] << " "; 
    // }
    // std::cout << std::endl;
    // MPI_Barrier(MPI_COMM_WORLD);

    double* horizontal_result = new double[n1 * n3];
    MPI_Gather(vertical_result, n1 / p1 * n3 / p2 * p1 , MPI_DOUBLE, horizontal_result, n1 / p1 * n3 / p2 * p1, MPI_DOUBLE, 0, Vertical_Comm);

    // std::cout << "[" << ranks[0] << "," << ranks[1] << "]" << std::endl; 
    delete[] vertical_result;
    return horizontal_result;
}

bool check(const int& n1, const int& n2, const int& n3, const int& p1, const int& p2, const int& size) {
    if(n1 % p1 != 0) {
        std::cout << "Bad format: n1 isn't divisible by p1" << std::endl;
        return false;
    }
    if(n3 % p2 != 0) {
        std::cout << "Bad format: n2 isn't divisible by p2" << std::endl;
        return false;
    }
    return true;

}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int size, rank;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    int n1 = atoi(argv[1]);
    int n2 = atoi(argv[2]);
    int n3 = atoi(argv[3]);
    int p1 = atoi(argv[4]);
    int p2 = atoi(argv[5]); 


    if(check(n1, n2, n3, p1, p2, size)) {
    }
    Matrix A; 
    Matrix B;

    if(rank == 0) {
        A = Matrix(n1, n2);
        A.fill();
        B = Matrix(n2, n3);
        B.fill();
    }

    int my_ranks[2];
    MPI_Comm Grid_Comm = create_grid_of_process(size, rank, p1, p2);
    MPI_Cart_coords(Grid_Comm, rank, 2, my_ranks);

    MPI_Comm Horizontal_Comm = create_horizontal_comm(Grid_Comm);
    MPI_Comm Vertiacal_Comm = create_vertical_comm(Grid_Comm);
    
    send_pieces_A(A, size, my_ranks, n1, n2, p1, Vertiacal_Comm);

    MPI_Datatype COLUMN;
    get_new_datatype(n2, n3, p2, &COLUMN);
    send_pieces_B(B, n2, n3, size, my_ranks, p2, COLUMN, Horizontal_Comm);

    broadcast_A(A, n1, n2, p1, Horizontal_Comm);
    broadcast_B(B, n2, n3, p2, Vertiacal_Comm);

    // print_all(A, "Matrix A: ", my_ranks);
    // sleep(1);
    // print_all(B, "Matrix A: ", my_ranks);
    // sleep(1);

    double* C = mul_minor(A, B);

    // if(my_ranks[0] == 0 && my_ranks[1] == 1) {
    //     for(int i = 0; i < 4; i++) {
    //         std::cout << C[i] << " " <<  std::endl;
    //     }
    // }

    double* result = gather_full_matrix_at_null_process(C, n1, n3, p1, p2, my_ranks, Horizontal_Comm, Vertiacal_Comm);
    if(rank == 0) {
        for(int i = 0; i < n1; i++) {
            for(int j = 0; j < n3; j++) {
                std::cout << result[j + i * n3] << " "; 
            }
            std::cout << std::endl;
        }
    }
    MPI_Finalize();
}

