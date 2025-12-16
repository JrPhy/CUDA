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

## 二、Multiply
CPU
```C++
#include <cstdlib>
#include <ctime>

void multiply(int m, int n, int l, float *x, float *y, float *z) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            z[i][j] = 0;
            for (int k = 0; k < l; ++k) {
                z[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}


int main() {
    int m = 1 << 3, n = 1 << 5, l = 1 << 4;
    float x[m][n], y[n][k], z[m][l];

    for (int i = 0; i < m; i++) {
          for (int j = 0; j < n; j++) {
               x[i][j] = rand() % 10 + (rand() % 10)/100;
          }
    }
    for (int i = 0; i < n; i++) {
          for (int j = 0; j < k; j++) {
               y[i][j] = rand() % 10 + (rand() % 10)/100;
          }
    }
    multiply(n, x, y, z)
    return 0;
}
```
GPU
```C++
#include <cstdlib>
#include <ctime>

__global__ void multiply(int m, int n, int l, float *x, float *y, float *z, int tile_size) {
    // x: m x l, y: l x n, z: m x n
    __shared__ float sA[tile_size][tile_size], sB[tile_size][tile_size];

    int row = blockIdx.y * tile_size + threadIdx.y;
    int col = blockIdx.x * tile_size + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (l + tile_size - 1) / tile_size; ++t) {
        // 載入 x 的 tile 到 shared memory
        if (row < m && t * tile_size + threadIdx.x < l)
            sA[threadIdx.y][threadIdx.x] = x[row * l + t * tile_size + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        // 載入 y 的 tile 到 shared memory
        if (col < n && t * tile_size + threadIdx.y < l)
            sB[threadIdx.y][threadIdx.x] = y[(t * tile_size + threadIdx.y) * n + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // tile 內部相乘累加
        for (int k = 0; k < tile_size; ++k)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < m && col < n) z[row * n + col] = sum;
}

int main() {
    int m = 1 << 3, n = 1 << 5, l = 1 << 4;
    float x[m][l], y[l][n], z[m][n];

    for (int i = 0; i < m; i++) {
          for (int j = 0; j < l; j++) {
               x[i][j] = rand() % 10 + (rand() % 10)/100;
          }
    }
    for (int i = 0; i < l; i++) {
          for (int j = 0; j < n; j++) {
               y[i][j] = rand() % 10 + (rand() % 10)/100;
          }
    }
    int tile_size = 16;
    dim3 blockSize(tile_size, tile_size);
    dim3 gridSize((n + tile_size - 1) / tile_size, (m + tile_size - 1) / tile_size);

    multiply<<<numBlocks, blockSize>>>(m, n, l, x, y, z, tile_size);
    return 0;
}
```
