#include<iostream>
#define SIZE 18000*18000 / 8 
#define WIDTH 18000

double scalar_mul(double* vec_1, double* vec_2, int N) {
    double res = 0.0;
    for(int i = 0; i < N; i++) {
        res += vec_1[i] * vec_2[i]; 
    }
    return res;
}

int main(int argc, char* argv[]) {
    double* arr = new double[SIZE];
    double* vector = new double[WIDTH]; 

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    
    for(int i = 0; i < SIZE; i+=WIDTH) {
        scalar_mul(arr + i, vector, WIDTH);
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    printf("Time taken: %lf sec.\n", end.tv_sec-start.tv_sec + 0.000000001*(end.tv_nsec-start.tv_nsec));
    return 0;
}
