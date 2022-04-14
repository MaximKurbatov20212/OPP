#include <iostream>
#include <mpi/mpi.h>
#include <string.h>
#include <vector>
#include <cmath>
#include <unistd.h>
#define t 0.01

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int full_rank = 0;
    MPI_Allreduce(&rank, &full_rank, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    std::cout << full_rank << " ";
    MPI_Finalize();
}