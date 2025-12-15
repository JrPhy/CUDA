## 一、ADD
CPU
```C++
void add(int n, float *x, float *y, float *z) {
     for (int i = 0; i < n; i++) z[i] = x[i] + y[i];
}

int main() {
    int n = 1<<15;
    float x[n], y[n], z[n];

    for (int i = 0; i < n; i++) {
        x[i] = i;
        y[i] = i * 2;
    }
    add(n, x, y, z)
    return 0;
}
```
GPU
```C++
#include <iostream>

// Kernel function to add the elements of two arrays
__global__ void add(int n, float* x, float* y) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) y[tid] = x[tid] + y[tid];
}

int main(void) {
    int n = 1 << 15;
    float *x, *y;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, n * sizeof(float));
    cudaMallocManaged(&y, n * sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < n; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Run kernel on 1M elements on the GPU
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(n, x, y);

    cudaDeviceSynchronize();

    // Free memory
    cudaFree(x);
    cudaFree(y);
    return 0;
}
```
