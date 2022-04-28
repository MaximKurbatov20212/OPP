#include <iostream>
#include <mpi/mpi.h>
#include <memory>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define a 100000 
#define eps 0.0000001

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
    return 6 - Phi(x, y, z) * a;
}

bool is_between_layers(const InputData& input, const int& size, const int i) {

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
                if(z == 0 || y == 0 || x == 0 || x == (input.Dx - 1) || y == (input.Dy - 1) || z == (input.Dz - 1)) {
                    array[get_index(input, x, y, z)] = Phi(get_X(input, x), get_Y(input, y), get_Z(input, z));
                }
            }
        }
    }
    return array;
}

void print_vector(const InputData& input, double* arr, int N, int size, int rank) {
    const int layer_size = input.get_size() / size / (input.Dx * input.Dy);
    std::cout << "rank = " << rank << std::endl;
    sleep(rank);
    // if(rank == 0) {
        for(int z = 0; z < layer_size; z++) {
            for(int y = 0; y < input.Dy; y++) {
                for(int x = 0; x < input.Dx; x++) {
                    std::cerr << (arr + input.Dx * input.Dy * z)[y * input.Dx + x]  << " ";
                }
                std::cerr << std::endl;
            }
            std::cerr << std::endl;
        }
    // }
    // MPI_Barrier(MPI_COMM_WORLD);
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



    // if(rank == 0 && i == 1 && j == 1 && k == 1) {
    //     std::cout << " k == " << K << std::endl; 
    //     std::cout << "p == " << p << std::endl;
    //     std::cout << "Pi-1,j,k + Pi+1,j,k / (hx *hx) = " << (extend_layer[get_index(input, i - 1, j, k)] + extend_layer[get_index(input, i + 1, j, k)]) / (hx * hx) << std::endl;
    //     std::cout << "Pi,j-1,k - 2 Pijk + Pi,j+1,k / (hy *hy) = " << (extend_layer[get_index(input, i, j - 1, k)] - 2 * extend_layer[get_index(input, i, j, k)] + extend_layer[get_index(input,  i, j + 1, k)]) / (hy * hy) << std::endl;
    //     std::cout << "Pi,j,k-1 - 2 Pijk + Pi,j,k+1 / (hz *hz) = " << (extend_layer[get_index(input, i, j, k - 1 )] - 2 * extend_layer[get_index(input, i, j, k)] + extend_layer[get_index(input,  i, j, k + 1)]) / (hz* hz) << std::endl;
    //     std::cout << "K * expr == " << K * expr << std::endl;
    // }
     
    return K * expr;
}

void calc_inside(const InputData& input, double* extend_layer, const int& size, const int& rank) {
    const int layer_height = input.get_size() / size / (input.Dx * input.Dy);
    // const int start = rank * (input.get_size() / size / (input.Dx * input.Dy));
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
    // std::cout << "start = " << start << std::endl;
    // sleep(rank);
    for(int z = start; z < start + layer_size; z++) {
        for(int y = 0; y < input.Dx; y++) {
            for(int x = 0; x < input.Dy; x++) {
                if(rank == 0) {
                    // std::cout << extend_layer[(z - start) * (input.Dy * input.Dx) + y * input.Dx + x] << " - ";
                    // std::cout << old_layer[(z - start) * (input.Dy * input.Dx) + y * input.Dx + x] << " ";
                    // std::cout << (extend_layer[(z - start) * (input.Dy * input.Dx) + y * input.Dx + x] - old_layer[(z - start) * (input.Dy * input.Dx) + y * input.Dx + x]) << " " << std::endl; 
                }
                
                if(extend_layer[(z - start) * (input.Dy * input.Dx) + y * input.Dx + x] - old_layer[(z - start) * (input.Dy * input.Dx) + y * input.Dx + x] > max) {
                    max = extend_layer[(z - start) * (input.Dy * input.Dx) + y * input.Dx + x] - old_layer[(z - start) * (input.Dy * input.Dx) + y * input.Dx + x];
                }
            }
            if(rank == 0) {
                // std::cerr << std::endl;
            }
        }
        if(rank == 0) {
            // std::cerr << std::endl;
        }
    }
    // MPI_Barrier(MPI_COMM_WORLD);

    // std::cout << "rank == " << rank << "max == " << max << std::endl;
    double maxes[size];
    int displ[size];
    int recv_counts[size];
    for(int i = 0; i < size; i++) {
        recv_counts[i] = 1;
        displ[i] = i;
    }
    MPI_Allgatherv(&max, 1, MPI_DOUBLE, maxes, recv_counts, displ , MPI_DOUBLE, MPI_COMM_WORLD);

    max = INT32_MIN;
    for(int i = 0; i < size; i++) {
        if(maxes[i] > max) {
            max = maxes[i];
        }
    }
    // std::cout << "max == " << max << std::endl;
    return max;
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

    const int layer_size = input_data.get_size() / size;
    const int extend_layer_size = input_data.get_size() / size + get_additional_size(input_data, size, rank);

    double* extend_layer = new double[extend_layer_size]();
    double* old_layer = new double[layer_size]();

    send_layers_each_process(extend_layer + get_start(input_data, size, rank), size, rank, input_data);
    memcpy(old_layer, extend_layer + get_start(input_data, size, rank), layer_size * sizeof(double));
    int count = input_data.Dx * input_data.Dy;
    do {
        
        send_borders(extend_layer, extend_layer_size, size, rank, count, requests);

        calc_inside(input_data, extend_layer + get_start(input_data, size, rank), size, rank);
        
        get_wait(size, rank, requests);

        memcpy(old_layer, extend_layer + get_start(input_data, size, rank), layer_size * sizeof(double));
    } while(abs(max(input_data, old_layer, extend_layer + get_start(input_data, size, rank), size, rank)) > eps); 

    if(rank == 0) {
        print_vector(input_data, extend_layer + get_start(input_data, size, rank), layer_size, size, rank);
    }
    delete[] extend_layer;
    delete[] old_layer;
    MPI_Finalize();
    return 0;
}
