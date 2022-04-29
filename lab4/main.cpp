#include <iostream>
#include <mpi/mpi.h>
#include <memory>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

#define a 100000.0 
#define eps 0.0000000001

#define Dx 2.0 
#define Dy 2.0
#define Dz 2.0

struct InputData {
    const double Nx, Ny, Nz, x0, y0, z0;
    InputData(double Nx, double Ny, double Nz, double x0, double y0, double z0): Nx(Nx), Ny(Ny), Nz(Nz),
                                                                                 x0(x0), y0(y0), z0(z0) {}
    int get_size() const {
        return Nx * Ny * Nz;
    }
};

inline int get_index(InputData input, int x, int y, int z) {
    return z * input.Nz * input.Ny + y * input.Nz + x;
}

double get_Hx(const int& N) {
    return Dx / (N - 1);
}

double get_Hy(const int& N) {
    return Dy / (N - 1);
}

double get_Hz(const int& N) {
    return Dz / (N - 1);
}

const double Phi(const double& x, const double& y, const double& z) {
    return x * x + y * y + z * z;
}

double get_X(const InputData& input, const double& i) {
    return input.x0 + i * get_Hx(input.Nx);
}

double get_Y(const InputData& input, const double& i) {
    return input.y0 + i * get_Hy(input.Ny);
}

double get_Z(const InputData& input, const double& i) {
    return input.z0 + i * get_Hz(input.Nz);
}

double right_expression(const double& x, const double& y, const double& z) {
    return 6.0 - Phi(x, y, z) * a;
}

double* get_border_values(InputData input, const int& size) {
    double* array = new double[input.get_size()]();

    for(int z = 0; z < input.Nz; z++) {
        for(int y = 0; y < input.Ny; y++) {
            for(int x = 0; x < input.Nx; x++) {
                if(z == 0 || y == 0 || x == 0 || x == (input.Nz - 1) || y == (input.Ny - 1) || z == (input.Nz - 1)) {
                    array[get_index(input, x, y, z)] = Phi(get_X(input, x), get_Y(input, y), get_Z(input, z));
                }
            }
        }
    }
    return array;
}

void print_result(const InputData& input, double* arr, int size, int rank) {
    const int layer_size = input.get_size() / size / (input.Nz * input.Ny);
    sleep(rank);
    std::cout << "rank = " << rank << std::endl;

    for(int z = 0; z < layer_size; z++) {
        for(int y = 0; y < input.Ny; y++) {
            for(int x = 0; x < input.Nx; x++) {
                std::cerr << (arr + (int)(input.Nz * input.Ny * z))[int(y * input.Nz + x)]  << " ";
            }
            std::cerr << std::endl;
        }
        std::cerr << std::endl;
    }
    // MPI_Barrier(MPI_COMM_WORLD);
}

void send_layers_each_process(double* layer, const int& size, const int& rank, InputData input) {
    double* matrix_3d;
    if(rank == 0) {
        matrix_3d = get_border_values(input, size);
    }

    MPI_Scatter(matrix_3d, input.get_size() / size, MPI_DOUBLE, layer, input.get_size() / size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if(rank == 0) {
        memcpy(layer, matrix_3d, input.get_size() / size);
        delete [] matrix_3d;
    }
}

const int get_additional_size(const InputData& input, const int& size, const int& rank) {
    if(size == 1) return 0;

    if(rank == size - 1 || rank == 0) {
        return input.Nz * input.Ny; 
    }
    return 2 * input.Nz * input.Ny;
}

int get_start(const InputData& input, const int& size, const int& rank) {
    if(size == 1) return 0;
    if(rank == 0) return 0;
    return input.Nz * input.Ny;
}

double calc_terrible_func(InputData input, int rank, const double* extend_layer, const int& i, const int& j, const int& k) {
    double hx = get_Hx(input.Nx);
    double hy = get_Hy(input.Ny);
    double hz = get_Hz(input.Nz);

    double K = 1 / ((2 / (hx * hx)) + (2 / (hy * hy)) + (2 / (hz * hz)) + a);
    double p = right_expression(get_X(input, i), get_Y(input, j), get_Z(input, k));

    double expr = ((extend_layer[get_index(input, i - 1, j, k)] + extend_layer[get_index(input, i + 1, j, k)]) / (hx * hx)) +
                  ((extend_layer[get_index(input, i, j - 1, k)] + extend_layer[get_index(input, i, j + 1, k)]) / (hy * hy)) + 
                  ((extend_layer[get_index(input, i, j, k - 1)] + extend_layer[get_index(input, i, j, k + 1)]) / (hz * hz)) -
                  p;

    return K * expr;
}

// TODO z - ?
// Calucates all cells which is not border
void calc_inside(const InputData& input, double* extend_layer, double* old_layer, const int& size, const int& rank) {
    const int layer_height = input.get_size() / size / (input.Nz * input.Ny);

    for(int z = 1; z < layer_height - 1; z++) {
        // std::cerr << z << std::endl; 
        for(int y = 1; y < input.Ny - 1; y++) {
            for(int x = 1; x < input.Nx - 1; x++) {
                extend_layer[get_index(input, x, y, z)] = calc_terrible_func(input, rank, extend_layer, x, y, z);
            }
        }
    }
}


double max(InputData input, const double* old_layer, const double* extend_layer, const int& size, const int& rank) {
    double max = 0;
    const int layer_size = input.get_size() / size / (input.Nz * input.Ny);

    const int start = rank * (input.get_size() / size / (input.Nz * input.Ny));

    for(int z = start; z < start + layer_size; z++) {
        for(int y = 0; y < input.Ny; y++) {
            for(int x = 0; x < input.Nx; x++) {
                if(fabs(extend_layer[(int)((z - start) * (input.Ny * input.Nz) + y * input.Nz + x)] - old_layer[(int)((z - start) * (input.Ny * input.Nz) + y * input.Nz + x)]) > max) {
                    max = fabs(extend_layer[(int)((z - start) * (input.Ny * input.Nz) + y * input.Nz + x)] - old_layer[(int)((z - start) * (input.Ny * input.Nz) + y * input.Nz + x)]);
                }
            }
        }
    }

    double global_max;
    MPI_Allreduce(&max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return global_max;
}

void send_borders(double* extend_layer, const int& extend_layer_size, const int& size, const int& rank, const int& count, MPI_Request* requests) {
    if(rank != 0) {
        MPI_Isend(&extend_layer[count], count, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &requests[0]);
        MPI_Irecv(&extend_layer[0], count, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &requests[1]);
    }
    if(rank != size - 1) {
        MPI_Isend(&extend_layer[extend_layer_size - 2 * count], count, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &requests[2]);
        MPI_Irecv(&extend_layer[extend_layer_size - count], count, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &requests[3]);
    }
}

// TODO z - ?
void calc_borders(const InputData& input, double* extend_layer, double* old_layer, const int& size, const int& rank) {
    const int layer_height = input.get_size() / size / (input.Nz * input.Ny);

    const int z_arr[2] = {rank == 0 ? 1 : 0 , (rank == size - 1)? layer_height - 2 : layer_height - 1};

    for(int i; i < 2; i++) {
        int z = z_arr[i];
        for(int y = 1; y < input.Ny - 1; y++) {
            for(int x = 1; x < input.Nx - 1; x++) {
                extend_layer[get_index(input, x, y, z)] = calc_terrible_func(input, rank, extend_layer, x, y, z);
            }
        }
    }
}

void get_wait(const int& size, const int& rank, MPI_Request* requests) {
    if(rank != 0){
        MPI_Wait(&requests[0], MPI_STATUS_IGNORE);
        MPI_Wait(&requests[1], MPI_STATUS_IGNORE);
    }
    if(rank != size - 1){
        MPI_Wait(&requests[2], MPI_STATUS_IGNORE);
        MPI_Wait(&requests[3], MPI_STATUS_IGNORE);
    }
}

double* calc_cube(const InputData& input_data, const int& size, const int& rank) {
    const int layer_size = input_data.get_size() / size;
    const int extend_layer_size = input_data.get_size() / size + get_additional_size(input_data, size, rank);

    double* extend_layer = new double[extend_layer_size]();
    double* old_layer = new double[layer_size]();

    int count = input_data.Nx * input_data.Ny;

    MPI_Request requests[4];

    // std::cerr << "HELLO" << std::endl;
    send_layers_each_process(extend_layer + get_start(input_data, size, rank), size, rank, input_data);

    do {
        memcpy(old_layer, extend_layer + get_start(input_data, size, rank), layer_size * sizeof(double));

        send_borders(extend_layer, extend_layer_size, size, rank, count, requests);

        calc_inside(input_data, extend_layer + get_start(input_data, size, rank), old_layer, size, rank);
        
        get_wait(size, rank, requests);

        calc_borders(input_data, extend_layer + get_start(input_data, size, rank), old_layer, size, rank);

    } while(max(input_data, old_layer, extend_layer + get_start(input_data, size, rank), size, rank) > eps); 

    delete [] extend_layer;

    return old_layer;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    InputData input = InputData(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), 
                                     atoi(argv[4]), atoi(argv[5]), atoi(argv[6]));

    double* result = calc_cube(input, size, rank);    

    // print_result(input, result, size, rank);    const int layer_size = input.get_size() / size / (input.Nz * input.Ny);
    const int layer_size = input.get_size() / size / (input.Nx * input.Ny);

    sleep(rank);
    // std::cout << "rank = " << rank << std::endl;
    
    for(int z = 0; z < layer_size; z++) {
        for(int y = 0; y < input.Ny; y++) {
            for(int x = 0; x < input.Nx; x++) {
                std::cerr << (result + (int)(input.Nx * input.Ny * z))[int(y * input.Nz + x)] - Phi(get_X(input, x), get_Y(input, y), get_Z(input, z + layer_size * rank ))  << " ";
            }
            std::cerr << std::endl;
        }
        std::cerr << std::endl;
    }



    delete[] result;
    MPI_Finalize();
    return 0;
}
