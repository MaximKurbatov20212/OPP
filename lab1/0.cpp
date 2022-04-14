#include <iostream>
#include <mpi/mpi.h>

void print(int* arr, int N, int size, int rank) {
    for(int i = 0; i < size; i++){
        if(rank == i) {
            std::cout << "rank = " << rank << std::endl;
            for(int i = 0; i < N; i++) {
                std::cout << arr[i];
            }
            std::cout << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

int main(int argc, char **argv){
    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 10;
    int* arr = new int(rank);

    int* all_vectors = new int[N](); // матрица строк которые надо сложить
    int* arr_shifts = new int[N]();    // интервалы, через которые надо записывать вектор к каждого процесса
    int* arr_size = new int[N];

    for(int i = 0; i < N; i++) {
        arr_shifts[i] = i;
        arr_size[i] = 1;
    }

    MPI_Allgatherv(arr, 1, MPI_INT, all_vectors, arr_size, arr_shifts, MPI_INT, MPI_COMM_WORLD); 

    // std::cout << *arr; 
    int sum = 0;
    for (size_t i = 0; i < size; i++)
    {
        sum += all_vectors[i];
        /* code */
    }
    
    std::cout << sum; 


    MPI_Finalize();
}