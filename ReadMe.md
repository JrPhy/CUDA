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
    for (int i = 0; i < n; i++)
        y[i] = x[i] + y[i];
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
    add<<<1, 1>>>(n, x, y);

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
其中 ```cudaMallocManaged``` 是分配記憶體的一種方式，分配一個統一記憶體 **Unified memory** 讓 CPU 與 GPU 都能存取，並且在 run-time 可以自動搬運記憶體，就不需要在寫程式時另外寫，但是速度也較慢，另一個則是 ```cudaMalloc``` 要搭配 ```cudaMemcpy``` 讓資料在 CPU 與 GPU 間搬運，效能也較好，一般也都是使用這兩個。```<<<...>>>``` 標示符是指用多少個 GPU Core，晚點來看如何分配。再來則是等待所有 GPU 跑完後把資料丟回 CPU 的 ```cudaDeviceSynchronize()```，最後就是去跑```cudaFree```來返還申請的記憶體。接著改用 ```cudaMalloc``` 要搭配 ```cudaMemcpy``` 來改寫以上例子，cudaMalloc 只分配記憶體在 GPU 上，所以需要用 cudaMemcpy 來把值複製到 GPU。

## 2. 如何分配給 GPU 
在 CPU 運算時很常使用多線程，而 GPU 可以開啟更多線程(thread)去做非常簡單的計算，所以通常在做加法時就可以丟給 GPU。<<<numBlocks, blockSize>>> 這就是用來告訴 GPU 要使用多少個，稱為 grid。所以通常會根據向量維度來做分配。以一維為例，就是 1* n 或是 n*1 的陣列，所以此例子中有 1 個 block，裡面有 256 個線程去跑。
```
int blockSize = 256;
int numBlocks = (n + blockSize - 1) / blockSize;
add<<<numBlocks, blockSize>>>(n, d_x, d_y);
```
```
dim3 blockSize(16, 16);  // 每個 block 有 16×16 threads
dim3 numBlocks((width+15)/16, (height+15)/16);
addMatrix<<<numBlocks, blockSize>>>(d_mat, width, height);
```
在多線程時最怕就是 race condition，如果要 block 中的 thread 共享資料，那就要都算完再去讀取，可以在 grid 中使用 ```__syncthreads()```

```c++
#include <iostream>
#include <math.h>

// Kernel function to add the elements of two arrays
__global__ void add(int n, float* x, float* y) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) y[tid] = x[tid] + y[tid];
}

int main(void) {
    int n = 1 << 15;  // 32768
    size_t size = n * sizeof(float);

    // Host memory allocation
    float *h_x = new float[n];
    float *h_y = new float[n];

    // Initialize host arrays
    for (int i = 0; i < n; i++) {
        h_x[i] = 1.0f;
        h_y[i] = 2.0f;
    }

    // Device memory allocation
    float *d_x, *d_y;
    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);

    // Copy data from host to device
    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);

    // Run kernel on n elements (use enough threads)
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(n, d_x, d_y);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost);

    // Check results (all values should be 3.0f)
    for (int i = 0; i < 10; i++) {  // print first 10 for brevity
        std::cout << h_y[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);

    // Free host memory
    delete[] h_x;
    delete[] h_y;

    return 0;
}

```
