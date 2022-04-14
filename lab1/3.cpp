#include <iostream>
#include <vector>
#include <mpi/mpi.h>

int main(int argc, char **argv) {
    int size, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (size < 2) {
        std::cerr << "This demo requires at least 2 procs." << std::endl;
        MPI_Finalize();
        return 1;
    }

    int datasize = 2*size + 1;
    std::vector<int> data(datasize);

    /* break up the elements */
    int *counts = new int[size];
    int *disps  = new int[size];

    int pertask = datasize/size;
    for (int i=0; i<size-1; i++)
        counts[i] = pertask;
    counts[size-1] = datasize - pertask*(size-1);

    disps[0] = 0;
    for (int i=1; i<size; i++)
        disps[i] = disps[i-1] + counts[i-1];

    int mystart = disps[rank];
    int mycount = counts[rank];
    int myend   = mystart + mycount - 1;

    /* everyone initialize our data */
    for (int i=mystart; i<=myend; i++)
        data[i] = 0;

    int nsteps = size;
    for (int step = 0; step < nsteps; step++ ) {

        for (int i=mystart; i<=myend; i++)
            data[i] += rank;

        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       &(data[0]), counts, disps, MPI_INT, MPI_COMM_WORLD);

        if (rank == step) {
            std::cout << "Rank " << rank << " has array: [";
            for (int i=0; i<datasize-1; i++)
                std::cout << data[i] << ", ";
            std::cout << data[datasize-1] << "]" << std::endl;
        }
    }

    delete [] disps;
    delete [] counts;

    MPI_Finalize();
    return 0;
}
