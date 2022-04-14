#include <iostream>
#include <mpi/mpi.h>
#include <time.h>
#include <unistd.h>

#define DIM 2

void print_(int* my_ranks) {
    for(int i = 0; i < 2; i++) {
        std::cout << my_ranks[i] << " ";
    }
    std::cout << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int size, rank;
    int N = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int array[3]{rank,rank,rank};
    int recv_buf[3]{rank, rank, rank};

    int arr_dims[2]{2,2};
    int arr_period[2]{1,1};
    MPI_Comm My_Comm;
    MPI_Cart_create(MPI_COMM_WORLD, DIM, arr_dims, arr_period, 0, &My_Comm);

    int my_ranks[2]{0,0};
    MPI_Cart_coords(My_Comm, rank, 2, my_ranks);
    //print_(my_ranks);
    
    // MPI_Comm VERT_LINE_COMM;
    int remain_dims[2]{true, false};
    // MPI_Cart_sub(My_Comm, remain_dims, &VERT_LINE_COMM);
    // for(int i = 0; i < 3; i++) {
    //     std::cout <<  recv_buf[i] << " ";
    // }
    // std::cout << std::endl;
    sleep(1);

    // MPI_Scatter(array, 3, MPI_INT, recv_buf, 3, MPI_INT, 0, VERT_LINE_COMM);

    // std::cout << "["<< my_ranks[0] << "," << my_ranks[1] << "]: ";
    // for(int i = 0; i < 3; i++) {
    //     std::cout << recv_buf[i] << " ";
    // }
    // std::cout << std::endl;

    MPI_Comm HOR_LINE_COMM;
    remain_dims[0] = false;
    remain_dims[1] = true;
    MPI_Cart_sub(My_Comm, remain_dims, &HOR_LINE_COMM);

    MPI_Scatter(array, 3, MPI_INT, recv_buf, 3, MPI_INT, 0, HOR_LINE_COMM);

    std::cout << "["<< my_ranks[0] << "," << my_ranks[1] << "]: ";
    for(int i = 0; i < 3; i++) {
        std::cout << recv_buf[i] << " ";
    }
    std::cout << std::endl;
    MPI_Finalize();
}
