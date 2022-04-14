#include <stdio.h>
#include <stdlib.h>
#include <mpi/mpi.h>

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
 
    // Get the number of processes and check only 3 processes are used
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(size != 2)
    {
        printf("This application is meant to be run with 3 processes.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
 
    // Number of cells per column.
    const int CELLS_PER_COLUMN = 4;
    // Number of per row.
    const int CELLS_PER_ROW = 4;
 
    // Create the vector datatype
    MPI_Datatype column_not_resized;
    MPI_Type_vector(CELLS_PER_COLUMN, 2, CELLS_PER_ROW, MPI_INT, &column_not_resized);
 
    // Resize it to make sure it is interleaved when repeated
    MPI_Datatype column_resized;
    MPI_Type_create_resized(column_not_resized, 0, 2 * sizeof(int), &column_resized);
    MPI_Type_commit(&column_resized);
 
    // Get my rank and do the corresponding job
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    int my_column[CELLS_PER_COLUMN * 2];

    if(my_rank == 0)
    {
        // Declare and initialise the full array
        int full_array[CELLS_PER_COLUMN][CELLS_PER_ROW] {{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}};
 
        // Send the column
        MPI_Scatter(full_array, 1, column_resized, my_column, CELLS_PER_COLUMN * 2, MPI_INT, 0, MPI_COMM_WORLD);
    }
    else
    {
        // Receive the column
        MPI_Scatter(NULL, 1, column_resized, my_column, CELLS_PER_COLUMN * 2, MPI_INT, 0, MPI_COMM_WORLD);
    }
 
    printf("MPI process %d received column made of cells %d, %d, %d, %d, %d, %d, %d, %d\n", my_rank, my_column[0], my_column[1], my_column[2], my_column[3], my_column[4], my_column[5], my_column[6], my_column[7]);
 
    MPI_Finalize();
 
    return EXIT_SUCCESS;
}