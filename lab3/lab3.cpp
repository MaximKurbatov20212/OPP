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

    MPI_Comm new_comm;
    const int remain_dims[2]{true, false};
    MPI_Cart_sub(My_Comm, remain_dims, &new_comm);
    return My_Comm;
}

MPI_Datatype get_new_datatype(const int& lines, const int& columns, const int& p2) {
    MPI_Datatype COLUMN_NOT_RESIZED;
    MPI_Datatype COLUMN_RESIZED;
    MPI_Type_vector(lines, columns / p2, columns, MPI_DOUBLE, &COLUMN_NOT_RESIZED);
    MPI_Type_commit(&COLUMN_NOT_RESIZED);

    MPI_Type_create_resized(COLUMN_NOT_RESIZED, 0, (columns / p2) * sizeof(double), &COLUMN_RESIZED);
    MPI_Type_commit(&COLUMN_RESIZED);
    return COLUMN_RESIZED;
}

MPI_Comm create_horizontal_comm(MPI_Comm& Old_Comm) {
    MPI_Comm New_Comm;
    int remain_dims[2]{false, true};
    MPI_Cart_sub(Old_Comm, remain_dims, &New_Comm);
    return New_Comm;
}

MPI_Comm create_vertical_comm(MPI_Comm& Old_Comm) {
    MPI_Comm New_Comm;
    int remain_dims[2]{true, false};
    MPI_Cart_sub(Old_Comm, remain_dims, &New_Comm);
    return New_Comm;
}


bool check(const int& n1, const int& n2, const int& n3, const int& p1, const int& p2, const int& size) {
    if(n1 % p1 != 0) {
        std::cout << "Bad format: n1 isn't divisible by p1" << std::endl;
        return false;
    }
    if(n3 % p2 != 0) {
        std::cout << "Bad format: n3 isn't divisible by p2" << std::endl;
        return false;
    }
    if(size != p1 * p2) {
        std::cout << "Bad format: p1 * p2 != number of process" << std::endl;
        return false;
    }
    if(p1 > n1 || p2 > n3) {
        std::cout << "Bad format: p1 > n1 || p2 > n3" << std::endl;
        return false;
    }
    return true;
}

void print_result(const double* result, const int& n1, const int& n3, const int& rank) {
    if(rank == 0) {
        for(int i = 0; i < n1; i++) {
            for(int j = 0; j < n3; j++) {
                std::cout << result[j + i * n3] << " "; 
            }
            std::cout << std::endl;
        }
    }
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


    if(!check(n1, n2, n3, p1, p2, size)) {
        return 0;
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
    
    double start = MPI_Wtime();

    send_pieces_A(A, size, my_ranks, n1, n2, p1, Vertiacal_Comm);

    MPI_Datatype COLUMN = get_new_datatype(n2, n3, p2);

    send_pieces_B(B, n2, n3, size, my_ranks, p2, COLUMN, Horizontal_Comm);

    broadcast_A(A, n1, n2, p1, Horizontal_Comm);
    broadcast_B(B, n2, n3, p2, Vertiacal_Comm);

    double* C = mul_minor(A, B);

    double* result = gather_full_matrix_at_null_process(C, n1, n3, p1, p2, my_ranks, Horizontal_Comm, Vertiacal_Comm);

    std::cout << MPI_Wtime() - start << std::endl;
    print_result(result, n1, n3, rank);
    delete result;
    delete C;
    MPI_Finalize();
}