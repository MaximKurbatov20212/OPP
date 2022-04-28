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

const double Phi(const double& x, const double& y, double z) {
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

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int size, rank;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    InputData input = InputData(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]));

    MPI_Request requests[4];
    const double hx = get_Hx(input, Nx);
    const double hy = get_Hx(input, Ny);
    const double hz = get_Hx(input, Nz);


    for(int z = 0; z < input.Dz; z++) {
        for(int y = 0; y < input.Dy; y++) {
            for(int x = 0; x < input.Dx; x++) {
                std::cerr << Phi(get_X(input, x), get_Y(input, y), get_Z(input, z)) << " ";
            }
            std::cerr << std::endl;
        }
        std::cerr << std::endl;
    }

    MPI_Finalize();
    return 0;
}

