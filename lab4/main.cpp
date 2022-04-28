#include <iostream>
#include <mpi/mpi.h>
#include <memory>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define a 100000 
#define eps 0.0000001

#define Nx 300 
#define Ny 300 
#define Nz 300 

struct InputData{
    const double Dx, Dy, Dz, x0, y0, z0;
    InputData(double Dx, double Dy, double Dz, double x0, double y0, double z0): Dx(Dx), Dy(Dy), Dz(Dz),
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

const double Phi(const double& x, const double& y, const double& z) {
    return x * x + y * y + z * z;
}

double get_X(const InputData& input, const double& i) {
    return input.x0 + i * get_Hx(input, Nx);
}

double get_Y(const InputData& input, const double& i) {
    return input.y0 + i * get_Hy(input, Ny);
}

double get_Z(const InputData& input, const double& i) {
    return input.z0 + i * get_Hz(input, Nz);
}

double right_expression(const double& x, const double& y, const double& z) {
    return 6.0 - Phi(x, y, z) * a;
}

double* get_border_values(InputData input, const int& size) {
    double* array = new double[input.get_size()]();

    for(int z = 0; z < input.Dz; z++) {
        for(int y = 0; y < input.Dy; y++) {
            for(int x = 0; x < input.Dx; x++) {
                if(z == 0 || y == 0 || x == 0 || x == (input.Dx - 1) || y == (input.Dy - 1) || z == (input.Dz - 1)) {
                    array[get_index(input, x, y, z)] = Phi(get_X(input, x), get_Y(input, y), get_Z(input, z));
                }
            }
        }
    }
    return array;
}

void print_result(const InputData& input, double* arr, int size, int rank) {
    const int layer_size = input.get_size() / size / (input.Dx * input.Dy);
    sleep(rank);
    std::cout << "rank = " << rank << std::endl;

    for(int z = 0; z < layer_size; z++) {
        for(int y = 0; y < input.Dy; y++) {
            for(int x = 0; x < input.Dx; x++) {
                std::cerr << (arr + (int)(input.Dx * input.Dy * z))[int(y * input.Dx + x)]  << " ";
            }
            std::cerr << std::endl;
        }
        std::cerr << std::endl;
    }
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
        return input.Dx * input.Dy; 
    }

    return 2 * input.Dx * input.Dy;
}

int get_start(const InputData& input, const int& size, const int& rank) {
    if(size == 1) return 0;
    if(rank == 0) return 0;
    return input.Dx * input.Dy;
}

double calc_terrible_func(InputData input, int rank, const double* extend_layer, const int& i, const int& j, const int& k) {
    double hx = get_Hx(input, Nx);
    double hy = get_Hy(input, Ny);
    double hz = get_Hz(input, Nz);
    double K = 1 / ((2 / (hx * hx)) + (2 / (hy * hy)) + (2 / (hz * hz)) + a);
    double p = right_expression(get_X(input, i), get_Y(input, j), get_Z(input, k));

    double expr = ((extend_layer[get_index(input, i - 1, j, k)] + extend_layer[get_index(input, i + 1, j, k)]) / (hx * hx)) +
                  ((extend_layer[get_index(input, i, j - 1, k)] - 2 * extend_layer[get_index(input, i, j, k)] + extend_layer[get_index(input,  i, j + 1, k)]) / (hy * hy)) + 
                  ((extend_layer[get_index(input, i, j, k - 1)] - 2 * extend_layer[get_index(input, i, j, k)] + extend_layer[get_index(input,  i, j, k + 1)]) / (hz * hz)) -
                  p;

    return K * expr;
}


// Calucates all cells which is not border
void calc_inside(const InputData& input, double* extend_layer, const int& size, const int& rank) {
    const int layer_height = input.get_size() / size / (input.Dx * input.Dy);
    const int start = rank == 0 ? 1 : 0;
    int end = (rank == size - 1) ? layer_height - 1: layer_height;

    for(int z = start; z < end; z++) {
        for(int y = 1; y < input.Dy - 1; y++) {
            for(int x = 1; x < input.Dx - 1; x++) {
                extend_layer[get_index(input, x, y, z)] = calc_terrible_func(input, rank, extend_layer, x, y, z);
            }
        }
    }
}


double max(InputData input, const double* old_layer, const double* extend_layer, const int& size, const int& rank) {
    double max = INT32_MIN;
    const int layer_size = input.get_size() / size / (input.Dx * input.Dy);

    const int start = rank * (input.get_size() / size / (input.Dx * input.Dy));
    for(int z = start; z < start + layer_size; z++) {
        for(int y = 0; y < input.Dx; y++) {
            for(int x = 0; x < input.Dy; x++) {
                if(extend_layer[(int)((z - start) * (input.Dy * input.Dx) + y * input.Dx + x)] - old_layer[(int)((z - start) * (input.Dy * input.Dx) + y * input.Dx + x)] > max) {
                    max = extend_layer[(int)((z - start) * (input.Dy * input.Dx) + y * input.Dx + x)] - old_layer[(int)((z - start) * (input.Dy * input.Dx) + y * input.Dx + x)];
                }
            }
        }
    }

    double global_max = 0;
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

void get_wait(const int& size, const int& rank, MPI_Request* requests) {
    if(rank != 0){
        MPI_Wait(&requests[0], MPI_STATUS_IGNORE);
        MPI_Wait(&requests[1], MPI_STATUS_IGNORE);
    }
    if(rank != size  - 1){
        MPI_Wait(&requests[2], MPI_STATUS_IGNORE);
        MPI_Wait(&requests[3], MPI_STATUS_IGNORE);
    }
}

double* calc_cube(const InputData& input_data, const int& size, const int& rank) {
    const int layer_size = input_data.get_size() / size;
    const int extend_layer_size = input_data.get_size() / size + get_additional_size(input_data, size, rank);

    double* extend_layer = new double[extend_layer_size]();
    double* old_layer = new double[layer_size]();

    int count = input_data.Dx * input_data.Dy;
    MPI_Request requests[4];

    send_layers_each_process(extend_layer + get_start(input_data, size, rank), size, rank, input_data);

    do {

        send_borders(extend_layer, extend_layer_size, size, rank, count, requests);
        calc_inside(input_data, extend_layer + get_start(input_data, size, rank), size, rank);
        get_wait(size, rank, requests);
        memcpy(old_layer, extend_layer + get_start(input_data, size, rank), layer_size * sizeof(double));
        
    } while(abs(max(input_data, old_layer, extend_layer + get_start(input_data, size, rank), size, rank)) > eps); 

    delete [] extend_layer;

    return old_layer;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    InputData input_data = InputData(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]));
    double* result = calc_cube(input_data, size, rank);    

    print_result(input_data, result, size, rank);

    delete[] result;
    MPI_Finalize();
    return 0;
}
