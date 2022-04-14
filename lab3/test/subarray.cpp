#include <stdio.h>
#include <mpi/mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int size, rank;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double array[16]{1, 2, 3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16}; 
    double subarray[8]; 

    MPI_Datatype filetype; 
    int sizes[4], subsizes[4], starts[2]; 
    
    sizes[0] = 4; 
    sizes[1] = 4; 

    subsizes[0] = 4;
    subsizes[1] = 2; 

    starts[0] = 0;
    starts[1] = 0; 
    // starts[1] = rank * subsizes[1]; 

    MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &filetype);
    MPI_Type_commit(&filetype);

    if(rank == 0) {
        MPI_Send(array, 1, filetype, 1, 0, MPI_COMM_WORLD);
    }

    if(rank == 1) {
        MPI_Status stat;
        MPI_Recv(subarray, 1, filetype, 0, 0, MPI_COMM_WORLD, &stat);
        for(int i = 0; i < 8; i++) {
            printf("%f ", subarray[i]);
        }
    }
    MPI_Finalize();

}