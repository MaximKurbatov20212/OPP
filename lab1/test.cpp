#include <iostream>
int scalar_mul(double* vec_1, double* vec_2, int N) {
    int res = 0;
    for(int i = 0; i < N; i++) {
        res += vec_1[i] * vec_2[i]; 
    }
    return res;
}

int main() {
    double vec_1[] = {2,1};
    double vec_2[] = {0, 0.03};
    std::cout << scalar_mul(vec_1, vec_2, 4) +scalar_mul(vec_1 + 4, vec_2, 4) ;
}