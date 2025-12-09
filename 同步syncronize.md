在平行計算需要注意競爭條件(race condition)，cuda 中也提供了一些關於同步的函數，也就是會等待某個範圍內都計算好後在到下一條指令

1. cudaMemcpy: 在 CPU 與 GPU 間複製記憶體的值(隱式)
2. cudaDeviceSynchronize: 等待整个 GPU 完成所有任務
3. __syncthreads：保證同一個 block 內線程共享記憶體的一致性
4. cudaStreamSynchronize: 等待某個 stream 的所有操作完成。
5. cudaEventSynchronize: 等待某個事件完成，常用於跨 stream 同步。

前面兩個在一般情況下比較常見，後面兩個則是在**非同步操作**才會用到。在做計算時沒有等到全部同步完就去操作，那可能會得到非預期的結果。

## 1. cudaDeviceSynchronize
這會等 GPU 內的所有 BLOCK 算完後再去執行下一段程式碼，所以通常在呼叫 KERNEL 時後面都會搭配 ```cudaDeviceSynchronize()```，下方例子就是全都算完後才會去執行 cout。
```C++
__global__ void add(int n, float* x, float* y) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) y[tid] = x[tid] + y[tid];
}

int main(void) {
    ...
    add<<<numBlocks, blockSize>>>(n, x, y);
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    for (int i = 0; i < c; i++) std::cout << y[i] << " ";
    ...
    return 0;
}
```

## 2. __syncthreads
在同一個 block 中也有許多 thread，如果需要等所有 thread 計算完在做下一次計算，就會需要用到此函數。例如在解波動方程時 $\ \frac{\partial u(x, t)}{\partial t} = \frac{\partial u(x, t)}{\partial x} $可以把空間計算去做平行，但是在時間步就需要等所有個 BLOCK 算完再進行下一次迭代
```C++
__global__ void wave1D(float *u_prev, float *u_curr, float *u_next, int N, float c) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i > 0 && i < N-1) { // t+1
        u_next[i] = 2.0f * u_curr[i] - u_prev[i] +
                    c * (u_curr[i-1] - 2.0f * u_curr[i] + u_curr[i+1]);
    }

    __syncthreads();

    if (i > 0 && i < N-1) { // 把 t+1 的值放進 u_curr
        u_prev[i] = u_curr[i];
        u_curr[i] = u_next[i];
    }
}
```
另一個就是在 kernel 中使用 shared memory 就需要做同步，例如在做矩陣相乘時，同個 row 與 col 相乘時可以不用照順序，但是最後加總時要等該 row 與 col 算完才能更新，否則就有可能在其他元素尚未計算完成之前就被更新了資料。
```C++
__global__ void matrixMulTiled(float *A, float *B, float *C, int N, int TILE_SIZE) {
    __shared__ float As[TILE_SIZE][TILE_SIZE], Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // 每個 thread 載入 A 和 B 的一部分到 shared memory
        if (row < N && t*TILE_SIZE + threadIdx.x < N)
            As[threadIdx.y][threadIdx.x] = A[row*N + t*TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t*TILE_SIZE + threadIdx.y < N)
            Bs[threadIdx.y][threadIdx.x] = B[(t*TILE_SIZE + threadIdx.y)*N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < N && col < N) C[row*N + col] = sum;
}
```
## 3. 原子操作
在多線程的程式中除了強制同步外，在簡單的計算中也可使用[原子操作](https://github.com/JrPhy/Multiple_Thread/blob/main/%E4%B8%8A%E4%B8%8B%E6%96%87%E4%BA%A4%E6%8F%9B%E8%88%87%E5%8E%9F%E5%AD%90%E6%93%8D%E4%BD%9C.md)，cuda 則是提供了對應的函數
| 函數名稱     | 功能說明               |
|--------------|------------------------|
| atomicAdd    | 對變數進行加法操作     |
| atomicSub    | 對變數進行減法操作     |
| atomicExch   | 將變數交換為新值       |
| atomicMin    | 設定變數為最小值       |
| atomicMax    | 設定變數為最大值       |
| atomicInc    | 對變數進行遞增操作     |
| atomicDec    | 對變數進行遞減操作     |
| atomicCAS    | 比較並交換 (Compare-And-Swap) |
| atomicAnd    | 位元 AND 操作          |
| atomicOr     | 位元 OR 操作           |
| atomicXor    | 位元 XOR 操作          |

而所有 CUDA 原子操作都回傳舊值，而不是新值，這樣設計是為了讓程式能判斷操作前的狀態，其中的```*address```是新值的位置，```val```則是舊值。
```C++
#include <stdio.h>

__global__ void xorKernel(int *data) {
    int tid = threadIdx.x;
    atomicXor(data, 1 << tid);  // 把不同位元位置設為 1
}

int main() {
    int h_data = 0, *d_data;

    cudaMalloc((void**)&d_data, sizeof(int));
    cudaMemcpy(d_data, &h_data, sizeof(int), cudaMemcpyHostToDevice);

    xorKernel<<<1, 4>>>(d_data);

    cudaMemcpy(&h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    printf("結果: %d (二進位: %08b)\n", h_data, h_data);
    return 0;
}
```
