## 1. 從 CPU 到 GPU
在此比較只用 CPU 與 使用 GPU 的寫法
```C++
// CPU only
#include <iostream>
#include <vector>

void add(int n, float *x, float *y) {
     for (int i = 0; i < n; i++) y[i] = x[i] + y[i];
}

int main() {
    int n = 1<<15;
    float x[n], y[n];

    for (int i = 0; i < n; i++) {
        x[i] = i;
        y[i] = i * 2;
    }

    add(n, x, y)
    for (int i = 0; i < n; i++) std::cout << y[i] << " ";
    return 0;
}

```
若要使用 GPU，則需要在函數前加上 __global__ 標示符，讓 CUDA 編譯器去辨認該函數可以跑在 GPU 上並在 CPU 被呼叫。
```C++
#include <iostream>
#include <math.h>

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

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    for (int i = 0; i < c; i++) std::cout << y[i] << " ";

    // Free memory
    cudaFree(x);
    cudaFree(y);
    return 0;
}
```
其中 ```cudaMallocManaged``` 是分配記憶體的一種方式，分配一個統一記憶體 **Unified memory** 讓 CPU 與 GPU 都能存取，並且在 run-time 可以自動搬運記憶體，就不需要在寫程式時另外寫，但是速度也較慢，```<<<...>>>``` 標示符是指用多少個 GPU Core，。再來則是等待所有 GPU 跑完後把資料丟回 CPU 的 ```cudaDeviceSynchronize()```，最後就是去跑```cudaFree```來返還申請的記憶體。

## 2. 核函數
給 GPU 執行的函數稱為核函數(Kernal function)，且只能返回```void```，所以都要用傳指標的方式進函數中，而使用多少 GPU 線程則是寫在核函數中。有以下標示符來給 CPU 與 GPU 辨認
```
__global__: CPU 呼叫給 GPU 跑
__device__: GPU 呼叫給 GPU 跑
__host__: CPU 呼叫給 CPU 跑
```
大部分情況是使用 __global__ 修飾核函數，但如果有核函數呼叫另一個核函數，則是用 __device__ 修飾。

## 3. GPU 的分塊
GPU 中有層級關係，調用一次核函數就是一個 grid，一個 grid 中有多個 block，thread 就是最小單位。以```<<<2, 64>>>```為例關係如下圖
```
Grid
 ├── Block 0
 │    ├── Thread 0
 │    ├── Thread 1
 │    ├── Thread 2
 │    ├── ...
 │    └── Thread 63
 │
 └── Block 1
      ├── Thread 0
      ├── Thread 1
      ├── Thread 2
      ├── ...
      └── Thread 63
```
![img](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/grid-of-thread-blocks.png)
在分配時有可能 thread 數量超過陣列大小，所以還是會在函數中寫以下判斷來保證不超過 index。
```
int tid = blockIdx.x * blockDim.x + threadIdx.x;
if (tid < n) y[tid] = x[tid] + y[tid];
```
<<<numBlocks, blockSize>>> 這就是用來告訴 GPU 要使用多少個。以一維為例，就是 1* n 或是 n*1 的陣列，所以此例子中有 1 個 block，裡面有 256 個線程去跑。當然也可以用多維的方式去分配，dim3 就是分別對應 x, y, z，若沒寫則預設為 1。可以看到程式中並沒有 ```threadIdx.x + blockIdx.x * blockDim.x``` 相關變數，這三個是 CUDA 在 kernel 裡提供的內建變數，用來計算全局索引來達到平行計算，如果是多維就會有多個，當然計算條件也會有多個。
```
dim3 blockSize(16, 16);  // 每個 block 有 16×16 threads
dim3 numBlocks((width+15)/16, (height+15)/16);
addMatrix<<<numBlocks, blockSize>>>(d_mat, width, height);
```
當然 kernel 中的 tid 也需要跟著改
```C++
// 假設每張圖片大小為 width x height，batchSize 張圖片
// 輸入是 RGB，每個像素有 3 個通道 (R,G,B)

__global__ void batchGrayScaleKernel(
    unsigned char* input,   // 輸入影像 (batchSize * width * height * 3)
    unsigned char* output,  // 輸出灰階影像 (batchSize * width * height)
    int width, int height, int batchSize) {
    // 計算 thread 對應的座標
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // pixel x
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // pixel y
    int z = blockIdx.z * blockDim.z + threadIdx.z;  // 第幾張圖片 (batch index)

    if (x < width && y < height) {
        int idx = (y * width + x) * 3; // RGB index
        unsigned char r = input[idx + 0];
        unsigned char g = input[idx + 1];
        unsigned char b = input[idx + 2];
        unsigned char gray = (unsigned char)(0.299f*r + 0.587f*g + 0.114f*b);
        output[y * width + x] = gray;
    }
}

int main() {
    int width = 1024, height = 768;
    dim3 blockDim(16, 16);  // 每個 Block 有 16x16 threads
    dim3 gridDim((width+15)/16, (height+15)/16); 
    // Grid 的第三維 (z) 對應到第幾張圖片
    batchGrayScaleKernel<<<gridDim, blockDim>>>(d_input, d_output, width, height);
}
```
## 4. 手動分配給 GPU 
另一個則是 ```cudaMalloc``` 要搭配 ```cudaMemcpy``` ，手動將資料在 CPU 與 GPU 間搬運，效能也較好，一般也都是使用這兩個。計算前先將 CPU 的記憶體複製到 GPU，計算完後再丟回來即可，這部分已有現成函數可使用。
```c++
#include <cuda_runtime.h>
#include <iostream>

__global__ void batchGrayScaleKernel(
    unsigned char* input,   // 輸入影像 (batchSize * width * height * 3)
    unsigned char* output,  // 輸出灰階影像 (batchSize * width * height)
    int width, int height, int batchSize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // pixel x
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // pixel y
    int z = blockIdx.z * blockDim.z + threadIdx.z;  // 第幾張圖片 (batch index)

    if (x < width && y < height && z < batchSize) {
        int idx = (z * width * height + y * width + x) * 3; // RGB index
        unsigned char r = input[idx + 0];
        unsigned char g = input[idx + 1];
        unsigned char b = input[idx + 2];

        unsigned char gray = (unsigned char)(0.299f*r + 0.587f*g + 0.114f*b);

        output[z * width * height + y * width + x] = gray;
    }
}

int main() {
    int width = 1024, height = 768, batchSize = 10;
    size_t inputSize  = batchSize * width * height * 3 * sizeof(unsigned char);
    size_t outputSize = batchSize * width * height * sizeof(unsigned char);

    unsigned char* h_input  = new unsigned char[inputSize];
    unsigned char* h_output = new unsigned char[outputSize];

    unsigned char* d_input;
    unsigned char* d_output;
    cudaMalloc((void**)&d_input, inputSize);
    cudaMalloc((void**)&d_output, outputSize);

    cudaMemcpy(d_input, h_input, inputSize, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16, 1);  
    dim3 gridDim((width+15)/16, (height+15)/16, batchSize);

    batchGrayScaleKernel<<<gridDim, blockDim>>>(d_input, d_output, width, height, batchSize);

    cudaMemcpy(h_output, d_output, outputSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    delete[] h_input;
    delete[] h_output;

    std::cout << "Batch grayscale conversion done!" << std::endl;
    return 0;
}
```
## 5. Block 大小選擇
在 GPU 硬體設計上，一個 Block 通常只有 1024 個 threads，所以最多只能有 <<<1, 1024>>>。實務上 Block 大小可以選擇接近 warp 大小的數字及其整數倍，不宜太多或是太少，在計算上會較有效率

## 5. 程式碼樣板
CPU 呼叫 GPU 函數時，該函數前方要用```__global__```修飾，函數內部要注意 index 小於陣列大小。main 函數呼叫 KernelFunction 時
```C++
// Kernel definition
__global__ void KernelFunc(float A[N][N], float B[N][N],
float C[N][N]) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N) C[i][j] = A[i][j] + B[i][j];
}

int main() {
    ...
    // Kernel invocation
    cudaMalloc(/*...*/);
    cudaMemcpy(/*...*/, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    cudaMemcpy(/*...*/, cudaMemcpyDeviceToHost);
    cudaFree(/*...*/)
    ...
}
```
