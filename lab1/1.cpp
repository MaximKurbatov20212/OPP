#include <iostream>
#include <mpi/mpi.h>
        
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int size, rank, sum, sum2;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double* complete_vector = new double[3]();
    double* a = new double[3];

    a[0] = rank;
    a[1] = rank;
    a[2] = rank;

    int* arr = new int[3]();
    int* arr_size = new int[3]();

    for(int i = 0; i < size; i++) {
        arr_size[i] = 1;
        
        if(i != size - 1) {
            arr[i + 1] =  arr[i] + arr_size[i];
        }
    }

    // arr[0] = 0;
    // arr[1] = 1;
    // arr[2] = 2;


    MPI_Allgatherv(a, 1, MPI_DOUBLE, complete_vector, arr_size, arr, MPI_DOUBLE, MPI_COMM_WORLD);

    for(int i = 0; i < size; i++){
        if(rank == i ) {
            for(int i = 0; i < 3; i++) std::cout << arr[i];
            std::cout << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
}