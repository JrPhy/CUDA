CUDA (Compute Unified Device Architecture，統一計算架構) 是 NVIDIA 研發的平行運算平台及編程模型，可利用繪圖處理單元 (GPU) 的能力大幅提升運算效能。原本是拿來做繪圖運算的，其實就是矩陣運算，而現在神經網路也是利用矩陣計算，所以對於 GPU 需求特別高。一般程式所使用的記憶體是在 CPU 與 RAM 中，而 cuda 的變數就會多了在 GPU 與 GPU DRAM(GRAM) 中，所以就會需要將變數從 CPU 搬到 GPU。

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
    for (int i = 0; i < c; i++) std::cout << y[i] << " ";
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

## 2. GPU 的分塊
GPU 中有層級關係，起調用一次函數就是一個 grid，一個 grid 中有多個 block，一個 block 中有多個 thread，thread 就是最小單位，其關係如下圖
```
Grid
└─ Block (0)
   ├─ Thread (0,0)
   ├─ Thread (0,1)
   ├─ Thread (0,2)
   └─ ...
└─ Block (1)
   ├─ Thread (1,0)
   ├─ Thread (1,1)
   ├─ Thread (1,2)
   └─ ...
└─ Block (2)
   ├─ Thread (2,0)
   ├─ Thread (2,1)
   ├─ Thread (2,2)
   └─ ...
```
在分配時有可能 thread 數量超過陣列大小，所以還是會在函數中寫以下判斷來保證不超過 index。
```
int tid = threadIdx.x + blockIdx.x * blockDim.x;
if (tid < n) y[tid] = x[tid] + y[tid];
```
<<<numBlocks, blockSize>>> 這就是用來告訴 GPU 要使用多少個。以一維為例，就是 1* n 或是 n*1 的陣列，所以此例子中有 1 個 block，裡面有 256 個線程去跑。當然也可以用多維的方式去分配，dim3 就是分別對應 x, y, z，若沒寫則預設為 1。
```
dim3 blockSize(16, 16);  // 每個 block 有 16×16 threads
dim3 numBlocks((width+15)/16, (height+15)/16);
addMatrix<<<numBlocks, blockSize>>>(d_mat, width, height);
```
當然 kernel 中的 tid 也需要跟著改
```
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
## 2. 手動分配給 GPU 
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
