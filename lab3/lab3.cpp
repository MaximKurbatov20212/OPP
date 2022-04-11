#include <iostream>
#include <mpi/mpi.h>
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
    return new_comm;
}

// TODO:
// implement send_pieces_B();
// distribute_piece();
// mul_minors();
// gather_full_matrix_at_null_procees();

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int size, rank;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int p1 = 2;
    int p2 = 2;

    int n1 = atoi(argv[1]);
    int n2 = atoi(argv[2]);
    int n3 = atoi(argv[3]);

    Matrix A; 
    // MPI_Datatype B_TYPE;
    // MPI_Type_vector(n2, n3, get_start_number(size, rank, n3), MPI_DOUBLE,  &B_TYPE);

    if(rank == 0) {
        A = Matrix(n1, n2);
        A.fill();
    }
    
    send_pieces_A(A, size, rank, n1, n2);

    MPI_Comm GRIG_COMM = create_grid_of_process(size, rank, p1, p2);

    print_all(A, "Matrix A: ", rank);
    MPI_Finalize();
}

