#include <stdio.h>
#include <mpi/mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int size, rank;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double array[2][2]{{0, 1}, {2, 3}}; 
    double subarray[2][1]; 

    MPI_Datatype filetype; 
    int sizes[2], subsizes[2], starts[2]; 
    
    sizes[0] = 2; 
    sizes[1] = 2; 

    subsizes[0] = 2;
    subsizes[1] = 1; 

    starts[0] = 0;
    starts[1] = 1; 
    // starts[1] = rank * subsizes[1]; 

    MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &filetype);
    MPI_Type_commit(&filetype);

    if(rank == 0) {
        MPI_Send(array, 1, filetype, 1, 0, MPI_COMM_WORLD);
    }

    if(rank == 1) {
        MPI_Status stat;
        MPI_Recv(subarray, 1, filetype, 0, 0, MPI_COMM_WORLD, &stat);
        for(int i = 0; i < 2; i++) {
            printf("%f ", subarray[0][i]);
        }
    }
    MPI_Finalize();

}