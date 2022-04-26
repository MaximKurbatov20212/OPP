#include <iostream>
#include <mpi/mpi.h>
#include <memory>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define a 1
#define eps 1

#define Nx 2
#define Ny 2
#define Nz 2

struct InputData{
    const int Dx, Dy, Dz, x0, y0, z0;
    InputData(int Dx, int Dy, int Dz, int x0, int y0, int z0): Dx(Dx), Dy(Dy), Dz(Dz),
                                                                x0(x0), y0(y0), z0(z0) {}
    int get_size() const {
        return Dx * Dy * Dz;
    }
};

inline int get_index(InputData input, int x, int y, int z) {
    return z * input.Dx * input.Dy + y * input.Dx + x;
}

double get_Hx(const InputData& input, const int& N) {
    return input.Dx / (N - 1);
}

double get_Hy(const InputData& input, const int& N) {
    return input.Dy / (N - 1);
}

double get_Hz(const InputData& input, const int& N) {
    return input.Dz / (N - 1);
}

const double Phi(const int& x, const int& y, int z) {
    return x * x + y * y + z * z;
}

double get_X(const InputData& input, const int& i) {
    return input.x0 + i * get_Hx(input, Nx);
}

double get_Y(const InputData& input, const int& i) {
    return input.y0 + i * get_Hy(input, Ny);
}

double get_Z(const InputData& input, const int& i) {
    return input.z0 + i * get_Hz(input, Nz);
}

double right_expression(const int& x, const int& y, const int& z) {
    return 6 * a - Phi(x, y, z);
}

bool between_layers(const InputData& input, const int& size, const int i) {

    const int layer_height = input.get_size() / size / (input.Dx * input.Dy);

    for(int j = 0; j < input.Dz; j += layer_height) {
        if(j == i || j - 1 == i) {
            return true;
        }
    }
    return false;
}

double* get_border_values(InputData input, const int& size) {
    double* array = new double[input.get_size()]();

    for(int z = 0; z < input.Dz; z++) {
        for(int y = 0; y < input.Dy; y++) {
            for(int x = 0; x < input.Dx; x++) {
                if(z == 0 || y == 0 || x == 0 || x == (input.Dx - 1) || y == (input.Dy - 1) || z == (input.Dz - 1) || between_layers(input, size, z)) {
                    // array[get_index(input, i, j, k)] = Phi(get_X(input, i), get_Y(input, j), get_Z(input, k));
                    array[get_index(input, x, y, z)] = 1;
                }
            }
        }
    }
    return array;
}

void print_vector(const InputData& input, double* arr, int N, int size, int rank) {
    sleep(rank);
    const int layer_size = input.get_size() / size / (input.Dx * input.Dy);
    std::cout << "rank = " << rank << std::endl;

    for(int z = 0; z < layer_size; z++) {
        for(int x = 0; x < input.Dx; x++) {
            for(int y = 0; y < input.Dy; y++) {

                std::cerr << (arr + input.Dx * input.Dy * z)[y * input.Dx + x]  << " ";
            }
            std::cerr << std::endl;
        }
        std::cerr << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void send_layers_each_process(double* layer, int& size, int& rank, InputData input) {
    double* matrix_3d;
    if(rank == 0) {
        matrix_3d = get_border_values(input, size);
        
        // for(int i = 0; i < 8; i++) {
        //     std::cerr << matrix_3d[i] << " ";
        // }
    }

    MPI_Scatter(matrix_3d, input.get_size() / size, MPI_DOUBLE, layer, input.get_size() / size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if(rank == 0) {
        memcpy(layer, matrix_3d, input.get_size() / size);
        delete matrix_3d;
    }
}

const int get_additional_size(const InputData& input, const int& size, const int& rank) {
    if(size == 1) return 0;

    if(rank == size - 1 || rank == 0) {
        return input.Dx * input.Dy; 
    }

    return 2 * input.Dx * input.Dy;
}

int get_start(const InputData& input, const int& size, const int& rank) {
    if(size == 1) return 0;
    if(rank == 0) return 0;
    return input.Dx * input.Dy;
}

double calc_terrible_func(InputData input, const double* extend_layer, const int& i, const int& j, const int& k) {
    double hx = get_Hx(input, Nx);
    double hy = get_Hy(input, Ny);
    double hz = get_Hz(input, Nz);
    double K = 1 / ((2 / hx * hx) + (2 / hy * hy) + (2 / hz * hz));
    double p = right_expression(get_X(input, i), get_Y(input, j), get_Z(input, k));

    double expr = ((extend_layer[get_index(input, i - 1, j, k)] + extend_layer[get_index(input, i - 1, j, k)]) / hx * hx) +
                  ((extend_layer[get_index(input, i, j - 1, k)] - 2 * extend_layer[get_index(input, i, j, k)] + extend_layer[get_index(input,  i, j - 1, k)]) / hy * hy) + 
                  ((extend_layer[get_index(input, i, j, k - 1)] - 2 * extend_layer[get_index(input, i, j, k)] + extend_layer[get_index(input,  i, j, k + 1)]) / hy * hy) -
                  p;

    return K * expr;
}

void calc_inside(const InputData& input, double* extend_layer, const int& size, const int& rank) {
    const int layer_height = input.get_size() / size / (input.Dx * input.Dy);
    int start = rank == 0 ? 0 : 1;
    int end = (rank == size - 1) ? layer_height : layer_height - 1;

    for(int z = start; z < end; z++) {
        for(int y = 1; y < input.Dy - 1; y++) {
            for(int x = 1; x < input.Dx - 1; x++) {
                extend_layer[get_index(input, x, y, z)] = calc_terrible_func(input, extend_layer, x, y, z);
                // extend_layer[get_index(input, x, y, z)] = 9; 
            }
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int size, rank;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    InputData input_data = InputData(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]));

    MPI_Request requests[4];
    const double hx = get_Hx(input_data, Nx);
    const double hy = get_Hx(input_data, Ny);
    const double hz = get_Hx(input_data, Nz);


    const int extend_layer_size = input_data.get_size() / size + get_additional_size(input_data, size, rank);
    // std::cout << "rank = " << rank << " size = " << extend_layer_size << std::endl;
    double* extend_layer = new double[extend_layer_size]();

    send_layers_each_process(extend_layer + get_start(input_data, size, rank), size, rank, input_data);
    const int layer_size = input_data.get_size() / size;
    // print_vector(input_data, extend_layer + get_start(input_data, size, rank), layer_size, size, rank);
    // print_vector(extend_layer, extend_layer_size, size, rank);

    int count = input_data.Dx * input_data.Dy;
    do {
        if(rank != 0) {
            MPI_Isend(&extend_layer[count], count, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &requests[0]);
            MPI_Irecv(&extend_layer[0], count, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &requests[1]);
        }
        if(rank != size - 1) {
            MPI_Isend(&extend_layer[extend_layer_size - 2 * count], count, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &requests[2]);
            MPI_Irecv(&extend_layer[extend_layer_size - count], count, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &requests[3]);
        }

        calc_inside(input_data, extend_layer, size, rank);

        if(rank != 0){
            MPI_Wait(&requests[0], MPI_STATUS_IGNORE);
            MPI_Wait(&requests[1], MPI_STATUS_IGNORE);
        }
        if(rank != size  - 1){
            MPI_Wait(&requests[2], MPI_STATUS_IGNORE);
            MPI_Wait(&requests[3], MPI_STATUS_IGNORE);
        }

    print_vector(input_data, extend_layer + get_start(input_data, size, rank), layer_size, size, rank);

    } while(max() < eps); 



    MPI_Finalize();
    return 0;
}