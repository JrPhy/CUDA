文中圖片來自於 https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
## 1. 記憶體架構
CPU 跟 GPU 內都有各自的記憶體 Registers 和 L1/L2 Caches，而 GPU 內部還有自己的 VRAM 跟 Shared Memory。VRAM 如同電腦中的 RAM 一樣，是 GPU 的外設記憶體，速度較慢，使用 cudaMalloc 的就是在 VRAM，cudaMemcpy 也是使用這部分的記憶體。而 Shared Memory 則是 block 內共享，適合平行運算，在 kernel 內用 ```__shared__``` 宣告，可以加速運算。
![img](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/gpu-devotes-more-transistors-to-data-processing.png)
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
| **Shared Memory**| 同一 block threads| 快     | thread 間資料交換、快取          | Kernel 內變數加上 ```__shared__``` |
| **Global Memory**| 所有 threads      | 慢     | 大量輸入/輸出資料                | 外部傳入 kernel |
| **Constant Memory** | 所有 threads   | 快（只讀） | 演算法常數、固定參數          | main 函數外變數加上 ```__constant__``` |
| **Texture Memory**  | 所有 threads   | 快（特殊存取） | 圖像處理、卷積操作          | cudaCreateTextureObject |
| **Local Memory** | 單一 thread       | 慢（實際在 global） | register 溢出時使用     | 編譯器自動 |
## 2. 利用 Shared Memory 加速運算
以前面的矩陣乘法例子來說
```c++
#include <stdio.h>
#include <cuda_runtime.h>

// CUDA 核函數：矩陣乘法
__global__ void matrixMulShared(float *C, float *A, float *B, int n) {
    // Shared Memory：暫存 A、B 的區塊 (tile)
    __shared__ float As[block_size][block_size];
    __shared__ float Bs[block_size][block_size];

    // 計算 thread 在矩陣中的位置
    int row = blockIdx.y * block_size + threadIdx.y;
    int col = blockIdx.x * block_size + threadIdx.x;

    float value = 0;

    // 分段載入 A、B 的 tile
    for (int k = 0; k < (n + block_size - 1) / block_size; ++k) {
        // 載入 A 的 tile
        if (row < n && k * block_size + threadIdx.x < n)
            As[threadIdx.y][threadIdx.x] = A[row * n + k * block_size + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0;

        // 載入 B 的 tile
        if (col < n && k * block_size + threadIdx.y < n)
            Bs[threadIdx.y][threadIdx.x] = B[(k * block_size + threadIdx.y) * n + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0;

        __syncthreads(); // 確保 tile 載入完成

        // 計算部分乘積
        for (int n = 0; n < block_size; ++n)
            value += As[threadIdx.y][n] * Bs[n][threadIdx.x];

        __syncthreads(); // 確保所有 threads 完成計算
    }

    // 寫入結果到 Global Memory
    if (row < n && col < n)
        C[row * n + col] = value;
}

int main() {
    int n = 64, block_size = 16;
    int size = n * n * sizeof(float);

    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    // 分配 Host 記憶體
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    for (int i = 0; i < n * n; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(block_size, block_size);
    dim3 dimGrid((n + block_size - 1) / block_size, (n + block_size - 1) / block_size);

    matrixMulShared<<<dimGrid, dimBlock>>>(d_C, d_A, d_B, n);

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
在硬體上 CPU 與 GPU 間是利用 PCIe 來傳輸，速度會比 GPU 內部記憶體還慢，如果記憶體夠的話可以多將需要存取的數據放在 Registers 或是用 ```__shared__``` 修飾，可以用來加速運算。
```
+-------------------+                   +-------------------+
|       CPU         |                   |        GPU        |
|  (Host Compute)   |                   | (Device Compute)  |
+---------+---------+                   +---------+---------+
          |                                     |
          |  Host API calls (CUDA runtime)      |
          |  e.g., cudaMalloc, cudaMemcpy,      |
          |        kernel launches              |
          v                                     v
+-------------------+                   +-------------------+
|   Host (CPU) RAM  | <==== PCIe =====> |  Device (GPU) VRAM |
|  (Pageable/Pinned)|    (DMA xfer)     |   (Global Memory)  |
+-------------------+                   +-------------------+
          ^                                     ^
          |                                     |
          +----H2D memcpy                       +---- D2H memcpy
          |                                     |
          v                                     v
+-------------------+                   +-------------------+
|  Host Staging     |                   |  GPU Kernels      |
|  Buffers (pinned) | --- launch --->   |  (Threads/Blocks) |
+-------------------+                   +-------------------+
                                            __shared__
```
![img](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/memory-hierarchy.png)

有時會遇到不確定要多少 Shared Memory，或是想寫成變數而非 #define 時，可以前方加上 extern，例如
```c++
__shared__ float sdata[256];
extern __shared__ float sdata[];
```
第二種就是在 runtime 時決定大小，並在呼叫 kernel 時多傳一個參數進去，但無法寫成多維度的形式，只能用多維轉一維的表達式去改寫
```C++
__global__ void matrixMulShared(float *C, float *A, float *B, int n, int block_size) {
    // 動態共享記憶體：一次分配 A、B 的 tile
    extern __shared__ float shared[];
    float* As = shared;                                   // 前半段存 A 的 tile
    float* Bs = shared + block_size * block_size;         // 後半段存 B 的 tile

    // 計算 thread 在矩陣中的位置
    int row = blockIdx.y * block_size + threadIdx.y;
    int col = blockIdx.x * block_size + threadIdx.x;

    float value = 0;

    // 分段載入 A、B 的 tile
    for (int k = 0; k < (n + block_size - 1) / block_size; ++k) {
        // 載入 A 的 tile
        if (row < n && k * block_size + threadIdx.x < n)
            As[threadIdx.y * block_size + threadIdx.x] = A[row * n + k * block_size + threadIdx.x];
        else
            As[threadIdx.y * block_size + threadIdx.x] = 0;

        // 載入 B 的 tile
        if (col < n && k * block_size + threadIdx.y < n)
            Bs[threadIdx.y * block_size + threadIdx.x] = B[(k * block_size + threadIdx.y) * n + col];
        else
            Bs[threadIdx.y * block_size + threadIdx.x] = 0;

        __syncthreads(); // 確保 tile 載入完成

        // 計算部分乘積
        for (int n = 0; n < block_size; ++n) {
            value += As[threadIdx.y * block_size + n] * Bs[n * block_size + threadIdx.x];
        }

        __syncthreads(); // 確保所有 threads 完成計算
    }

    // 寫入結果到 Global Memory
    if (row < n && col < n)
        C[row * n + col] = value;
}

int main() {
    ...
    const int block_size = 32;
    dim3 block(block_size, block_size);
    dim3 grid((n + block_size - 1) / block_size, (n + block_size - 1) / block_size);
          
    // 需要分配兩個 tile (A、B)，所以大小是 2 * block_size * block_size
    size_t sharedMemSize = 2 * block_size * block_size * sizeof(float);
    // sharedMemSize 就是多傳進去的
    matrixMulShared<<<grid, block, sharedMemSize>>>(C, A, B, n);
    ...
}
```
當然共享記憶體也是有上限的，如果超過硬體上限則會有 run-time error，若是要存取同個 Bank，則會有 Bank Conflict。

## 4. Bank Conflict 
Shared Memory 雖然可以加速計算，但是其大小最大目前(2025)不超過 1024 KB，如果使用太多就有可能會讓多個 Block 去存取同一塊 Shared Memory。而共享記憶體被劃分成多個「bank」，每個 bank 可以同時處理一個 thread 的存取，目前 NVIDIA GPU 架構幾乎都是 32 個 bank，每個 bank 的寬度是 8 bytes。如果不同 threads 用到同一塊共享記憶體的不同位置時，那就須遵守先來後到的順序，就會變成***串行***而非並行，所以就要盡量避免使用到同一塊 bank，如下圖所示，只要沒有指向同一塊 bank 就不會有 Conflict。
![img](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/examples-of-irregular-shared-memory-accesses.png)\
如果都用 1D array 就幾乎不會遇到，但實務上很常用到 nD array，例如多張圖片，或是矩陣相乘等。而 padding 是一種很好的解法，以二維 32x32 矩陣相乘的例子
```C++
__global__ void sharedABMultiply(float *a, float* b, float *c,
                                 int N) // N = 32
{
    __shared__ float aTile[TILE_DIM][TILE_DIM],
                     bTile[TILE_DIM][TILE_DIM];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    aTile[threadIdx.y][threadIdx.x] = a[row*TILE_DIM+threadIdx.x];
    bTile[threadIdx.y][threadIdx.x] = b[threadIdx.y*N+col];
    __syncthreads();
    for (int i = 0; i < TILE_DIM; i++) {
        sum += aTile[threadIdx.y][i]* bTile[i][threadIdx.x];
    }
    c[row*N+col] = sum;
}
```
```
Thread 0 → data[0][0] → Bank 0
Thread 1 → data[0][1] → Bank 1
...
Thread 31 → data[0][31] → Bank 31
Thread 32 → data[1][0] → Bank 0 X Conflict
Thread 33 → data[1][1] → Bank 1 X Conflict
```
可以看到這樣就把 data[i][0], data[i][1], ..., data[i][31] 都打到同個 Bank 上，所以就會有等待的時間。所以在宣告 ```__shared__``` 時可以在每個 col+1，這樣就可以打亂
```c++
__shared__ float aTile[TILE_DIM][TILE_DIM+1],
                 bTile[TILE_DIM][TILE_DIM+1];
...
/*
Thread 0 → data[0][0] → Bank 0
Thread 1 → data[0][1] → Bank 1
...
Thread 31 → data[0][31] → Bank 31
Thread 32 → data[1][0] → Bank 1
Thread 33 → data[1][1] → Bank 2
*/
```
即便用一維陣列也是有相同問題，因為編譯後其實都是一維陣列，所以也可以用相同方法去避免。最理想的情況，shared 無 conflict 比 global 快 400–600 倍；即使有 conflict，仍比 global 快一個數量級。

