
#include <stdlib.h>
#include <mpi/mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int size, rank;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int send_buf[16]{0, 1, 2, 3,
                     4, 5, 6, 7,
                     8, 9, 10, 11, 
                     12, 13, 14, 15};

    int recv_buf[16];

    MPI_Datatype B_TYPE;
    MPI_Datatype B;
    MPI_Type_vector(4, 2, 4, MPI_INT, &B_TYPE);
    MPI_Type_commit(&B_TYPE);

    MPI_Type_create_resized(B_TYPE, 0, sizeof(int), &B);
    MPI_Type_commit(&B);

    MPI_Scatter(send_buf, 4, B, recv_buf, 4, B, 0, MPI_COMM_WORLD);
    
    // if(rank == 0) {
        // MPI_Send(send_buf, 2, B, 1, 0, MPI_COMM_WORLD);
    // }

    if(rank == 1) {
        // MPI_Status stat;
        // MPI_Recv(recv_buf, 2, B, 0, 0, MPI_COMM_WORLD, &stat);
        for(int i = 0; i < 16; i++) {
            printf("%d ", recv_buf[i]);
        }
    }

    MPI_Finalize();
}