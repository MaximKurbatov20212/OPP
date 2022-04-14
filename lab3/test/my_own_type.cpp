#include <iostream>
#include <mpi/mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int size, rank;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int* send_buf = new int[13]{0,1,2,3, 4,5,6,7,8, 9,10,11,12};
    int* recv_buf = new int[13]();

    MPI_Datatype B;
    MPI_Type_vector(3, 4, 4, MPI_INT, &B);
    MPI_Type_commit(&B);
    if(rank == 0) {
        MPI_Send(send_buf, 3, B, 1, 0, MPI_COMM_WORLD);
        MPI_Send(send_buf, 2, B, 2, 0, MPI_COMM_WORLD);
    }

    if(rank == 1) {
        MPI_Status stat;
        MPI_Recv(recv_buf, 3, B, 0, 0, MPI_COMM_WORLD, &stat);
        for(int i = 0; i < 12 ; i++) {
            std::cout << recv_buf[i] << " ";
        }
        std::cout << std::endl;
    }

    if(rank == 2) {
        MPI_Status stat;
        MPI_Recv(recv_buf, 2, B, 0, 0, MPI_COMM_WORLD, &stat);
        for(int i = 0; i < 8; i++) {
            std::cout << recv_buf[i] << " ";
        }
        std::cout << std::endl;
    }
    MPI_Finalize();
}