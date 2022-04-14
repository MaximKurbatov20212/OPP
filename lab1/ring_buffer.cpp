#include <iostream>
#include <mpi/mpi.h>
#include <string.h>
#include <cmath>
#include <unistd.h>

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

void send_piece_of_vector_next(double* piece_of_vector, double* piece_of_vector_2, int rank, int size, int N) {
    MPI_Status status;
    if(rank % 2 == 0) {
        MPI_Send(piece_of_vector, 4, MPI_DOUBLE, (rank + 1) % size, 0, MPI_COMM_WORLD);
        MPI_Recv(piece_of_vector, 4, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
    }
    else {
        MPI_Recv(piece_of_vector_2, 4, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        MPI_Send(piece_of_vector, 4, MPI_DOUBLE, (rank + 1) % size, 0, MPI_COMM_WORLD);
        memcpy(piece_of_vector, piece_of_vector_2, 4 * sizeof(double));
    }
}


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double piece[] = {rank,rank, rank, rank};

    print_vector(piece, 4, size, rank);
    double* piece2;
    if(rank % 2 == 1) {
        piece2 = new double[4]();
    }

    for(int i = 0; i < size; i++) {
        send_piece_of_vector_next(piece, piece2, rank, size, 4);
    }
    print_vector(piece, 4, size, rank);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Finalize();
}