CPU 跟 GPU 內都有各自的記憶體 Registers 和 L1/L2 Caches，而 GPU 內部還有自己的 VRAM 跟 Shared Memory。VRAM 如同電腦中的 RAM 一樣，是 GPU 的外設記憶體，速度較慢，使用 cudaMalloc 的就是在 VRAM，cudaMemcpy 也是使用這部分的記憶體。而 Shared Memory 則是 block 內共享，適合平行運算，在 kernel 內用 __shared__ 宣告，可以加速運算。
```
                 +----------------------+
                 |     Registers        |  <-- 每個 thread 專屬，速度最快
                 +----------------------+
                           |
                 +----------------------+
                 |   Shared Memory      |  <-- Block 內 threads 共享，延遲低
                 +----------------------+      kernel 內用 __shared__ 宣告
                           |
                 +----------------------+
                 |   L1 / L2 Cache      |  <-- 快取，加速存取
                 +----------------------+
                           |
                 +----------------------+
                 |   Global Memory      |  <-- VRAM (GDDR/HBM)，容量大但延遲高
                 +----------------------+      main 中用 cudaMalloc
                           |
                 +----------------------+
                 | Constant / Texture   |  <-- 特殊用途，只讀或影像優化
                 +----------------------+
```
在前面提到的[矩陣乘法](https://github.com/JrPhy/CUDA/blob/main/%E5%90%8C%E6%AD%A5syncronize.md#2-__syncthreads)就有用到 Shared Memory，當然也可以不用，但速度就比較慢，因為要每次去存取 Global Memory 的值。所以選擇使用的記憶體也是影響效能因素之一，下方整理出使用時機
# 程式設計時的使用場景

| 記憶體種類       | 存取範圍          | 速度   | 常見用途                         | 從何而來  |
|------------------|-------------------|--------|----------------------------------|-------|
| **Registers**    | 單一 thread       | 最快   | thread 私有變數                  | Kernel 內部直接宣告變數 |
| **Shared Memory**| 同一 block threads| 快     | thread 間資料交換、快取          | Kernel 內變數加上 __shared__ |
| **Global Memory**| 所有 threads      | 慢     | 大量輸入/輸出資料                | 外部傳入 kernel |
| **Constant Memory** | 所有 threads   | 快（只讀） | 演算法常數、固定參數          | main 函數外變數加上 __constant__ |
| **Texture Memory**  | 所有 threads   | 快（特殊存取） | 圖像處理、卷積操作          | cudaCreateTextureObject |
| **Local Memory** | 單一 thread       | 慢（實際在 global） | register 溢出時使用     | 編譯器自動 |

以前面的矩陣乘法例子來說
```c++
#include <stdio.h>
#include <cuda_runtime.h>

// CUDA 核函數：矩陣乘法
__global__ void matrixMulShared(float *C, float *A, float *B, int N) {
    // Shared Memory：暫存 A、B 的區塊 (tile)
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // 計算 thread 在矩陣中的位置
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float value = 0;

    // 分段載入 A、B 的 tile
    for (int k = 0; k < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++k) {
        // 載入 A 的 tile
        if (row < N && k * BLOCK_SIZE + threadIdx.x < N)
            As[threadIdx.y][threadIdx.x] = A[row * N + k * BLOCK_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0;

        // 載入 B 的 tile
        if (col < N && k * BLOCK_SIZE + threadIdx.y < N)
            Bs[threadIdx.y][threadIdx.x] = B[(k * BLOCK_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0;

        __syncthreads(); // 確保 tile 載入完成

        // 計算部分乘積
        for (int n = 0; n < BLOCK_SIZE; ++n)
            value += As[threadIdx.y][n] * Bs[n][threadIdx.x];

        __syncthreads(); // 確保所有 threads 完成計算
    }

    // 寫入結果到 Global Memory
    if (row < N && col < N)
        C[row * N + col] = value;
}

int main() {
    int N = 64, BLOCK_SIZE = 16;
    int size = N * N * sizeof(float);

    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    // 分配 Host 記憶體
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matrixMulShared<<<dimGrid, dimBlock>>>(d_C, d_A, d_B, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("C[0..9]: ");
    for (int i = 0; i < 10; i++)  printf("%.1f ", h_C[i]);
    printf("\n");

    // 釋放記憶體
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```
