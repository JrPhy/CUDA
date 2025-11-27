CUDA (Compute Unified Device Architecture，統一計算架構) 是 NVIDIA 研發的平行運算平台及編程模型，可利用繪圖處理單元 (GPU) 的能力大幅提升運算效能。原本是拿來做繪圖運算的，其實就是矩陣運算，而現在神經網路也是利用矩陣計算，所以對於 GPU 需求特別高。一般程式所使用的記憶體是在 CPU 與 RAM 中，而 cuda 的變數就會多了在 GPU 與 GPU DRAM(GRAM) 中，所以就會需要將變數從 CPU 搬到 GPU。

## 1. 向量相加
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
