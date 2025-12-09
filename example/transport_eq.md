CPU
```C++
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    const int  = 1000;       // 空間格點數
    const int steps = 1000;   // 時間步數
    const double c = 1.0;     // 傳輸速度
    const double dx = 1.0 / n;
    const double dt = 0.5 * dx / c; // CFL 條件

    std::vector<double> u(n, 0.0), u_new(n, 0.0);

    for (int i = 0; i < n; i++) {
        double x = i * dx;
        u[i] = exp(-100 * (x - 0.5) * (x - 0.5));
    }

    for (int n = 0; n < steps; n++) {
        for (int i = 1; i < n; i++) {
            u_new[i] = u[i] - c * dt / dx * (u[i] - u[i - 1]);
        }
        u = u_new;
    }

    for (int i = 0; i < n; i++) {
        std::cout << u[i] << "\n";
    }

    return 0;
}
```
CUDA
```C++
#include <iostream>
#include <cmath>

__global__ void advectionShared(double* u, double* u_new, int n, double c, double dt, double dx) {
    extern __shared__ double s_u[];  // 動態共享記憶體
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    if (i < n) s_u[tid] = u[i];

    __syncthreads();

    // 計算 (需要前一格，注意邊界)
    if (i > 0 && i < n) {
        double left = (tid == 0) ? u[i - 1] : s_u[tid - 1];
        u_new[i] = s_u[tid] - c * dt / dx * (s_u[tid] - left);
    }
}


int main() {
    const int n = 1000;
    const int steps = 1000;
    const double c = 1.0;
    const double dx = 1.0 / n;
    const double dt = 0.5 * dx / c;

    double* h_u = new double[n];
    double* h_u_new = new double[n];

    for (int i = 0; i < n; i++) {
        double x = i * dx;
        h_u[i] = exp(-100 * (x - 0.5) * (x - 0.5));
    }

    double *d_u, *d_u_new;
    cudaMalloc(&d_u, n * sizeof(double));
    cudaMalloc(&d_u_new, n * sizeof(double));
    cudaMemcpy(d_u, h_u, n * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    for (int n = 0; n < steps; n++) {
        advectionKernel<<<numBlocks, blockSize>>>(d_u, d_u_new, n, c, dt, dx);
        cudaMemcpy(d_u, d_u_new, n * sizeof(double), cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy(h_u, d_u, n * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++)
        std::cout << h_u[i] << "\n";

    cudaFree(d_u);
    cudaFree(d_u_new);
    delete[] h_u;
    delete[] h_u_new;

    return 0;
}

```
