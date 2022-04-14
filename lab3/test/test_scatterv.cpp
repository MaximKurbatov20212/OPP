#include <iostream>
#include <mpi/mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int size, rank;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int* recv_buf = new int[2]();
    int* send_counts = new int[4]{2,1,1,1};
    int* recv_counts = new int[4]{2,1,1,1};

    int* arr = new int[5]{3,2,1,0,-1};
    int* displ = new int[4]{0,2,3,4};
    
    MPI_Scatterv(arr, send_counts, displ, MPI_INT, recv_buf, recv_counts[rank], MPI_INT, 0, MPI_COMM_WORLD);

    std::cout << rank << ": " << recv_buf[0] << " " << recv_buf[1] << std::endl;

    MPI_Finalize();
}