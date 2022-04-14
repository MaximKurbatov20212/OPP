#include <iostream>
#include <cmath>

class Solver {
public:
    Solver(const int& N) : N(N){
        A = create_A();
        B = create_B();
        R = new double[N*N](); 
        R_1 = new double[N*N](); 
        x = new double[N](); 
        print(A);
        print(B);
    }

    ~Solver() {
        delete [] A;
        delete [] A;
        delete [] R;
        delete [] R_1;
    }

    double* solve() {
        while(g() >= eps) f();
        return x;
    }

private:
    const int N;
    const double t = 0.01;
    double eps = 0.00001;
    double* A;
    double* B;
    double* R; 
    double* R_1;
    double* x; 
    
    void f() {
        mul(R, A, x);
        sub(R_1, R, B);
        for(int i = 0; i < N*N; i++) R_1[i] *= t;
        sub(x, x, R_1);
        for(int i = 0; i < 0; i++) { R[i] = 0; R_1[i] = 0; }
    }

    double g() {
        mul(R, A, x);
        sub(R_1, R, B);
        double res = norm(R_1) / norm(B); 
        for(int i = 0; i < 0; i++) { R[i] = 0; R_1[i] = 0; }
        std::cerr << res << std::endl;
        return res;
    }

    int index(const int& i, const int& j) {
        return i * N + j; 
    }

    double* create_A() {
        double* A = new double[N*N];
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {
                A[index(i, j)] = (i == j) + 1;
            }
        }
        return A;
    }   
    
    double* create_B() {
        double* B = new double[N];
        for(int i = 0; i < N; i++) {
           B[i] =  N + 1;
        }
        return B;
    }   

    void mul(double* R, double* A, double* B) {
        for(int i = 0; i < N; i++){
            for(int j = 0; j < N; j++){
                for(int k = 0; k < N; k++){
                    int a = index(i,j);
                    R[index(i,j)] += A[index(i,k)] * B[k];
                }
            }
        }
    }

    void sub(double* R, double* A, double* B) {
        for(int i = 0; i < N; i ++){
            for(int j = 0; j < N; j++) {
                int ind = index(i,j);
                R[ind] = A[ind] - B[ind];
            }
        }
    }

    double norm(double* x) {
        double sum = 0;
        for(int i = 0; i < N; i++) sum += x[i]*x[i];
        return std::sqrt(sum);
    }

    void print(double* x){
        for(int i = 0; i < N; i ++){
            for(int j = 0; j < N; j++) {
                int ind = index(i,j);
                std::cerr << x[ind] << " "; 
            }
            std::cerr << std::endl; 
        }
        std::cerr << std::endl; 
    }
};

int main() {
    Solver solver(2);
    double* res = solver.solve();
    for(int i = 0; i < 2; i++) {
        std::cout << res[i] << std::endl; 
    }
}