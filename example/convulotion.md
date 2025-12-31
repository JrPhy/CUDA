一、CPU
```C++
#include <vector>
std::vector<int> conv2D_1Darray(const std::vector<int>& img, int width, int height,
                                const std::vector<int>& kernel, int ksize) {
    int pad = ksize / 2;
    std::vector<int> result(img.size(), 0);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int sum = 0;
            for (int m = -pad; m <= pad; m++) {
                for (int n = -pad; n <= pad; n++) {
                    int x = i + m;
                    int y = j + n;
                    if (x >= 0 && x < height && y >= 0 && y < width) {
                        sum += img[x * width + y] * kernel[(m + pad) * ksize + (n + pad)];
                    }
                }
            }
            result[i * width + j] = sum;
        }
    }
    return result;
}

int main() {
    int width = 2880, height = 2800;
    std::vector<int> imgFlat(width * height, 1);                   // 測試用影像 (一維)
    std::vector<int> kernelFlat(25, 1);                            // 5x5 kernel (一維)

    std::vector<int> r3 = conv2D_1Darray(imgFlat, width, height, kernelFlat, 5);
    return 0;
}
```
二、GPU
```C++

#include <vector>
#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void conv2D_1Darray_cuda_shared(const int* img, int width, int height,
                                           const int* kernel, int ksize, int* result) {
    extern __shared__ int s_img[];
    int pad = ksize / 2;
    int tx = threadIdx.x, ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    // 計算 shared memory 的寬高
    int sm_width = blockDim.x + ksize - 1;
    int sm_height = blockDim.y + ksize - 1;

    // 對應到 global memory 的左上角座標
    int sm_x = x - pad;
    int sm_y = y - pad;

    // 將 block 對應區域搬到 shared memory
    for (int dy = ty; dy < sm_height; dy += blockDim.y) {
        for (int dx = tx; dx < sm_width; dx += blockDim.x) {
            int img_x = blockIdx.x * blockDim.x + dx - pad;
            int img_y = blockIdx.y * blockDim.y + dy - pad;
            int sm_idx = dy * sm_width + dx;
            if (img_x >= 0 && img_x < width && img_y >= 0 && img_y < height)
                s_img[sm_idx] = img[img_y * width + img_x];
            else
                s_img[sm_idx] = 0;
        }
    }
    __syncthreads();

    // 只讓合法的 thread 做卷積
    if (x < width && y < height) {
        int sum = 0;
        for (int m = 0; m < ksize; ++m) {
            for (int n = 0; n < ksize; ++n) {
                int sm_idx = (ty + m) * sm_width + (tx + n);
                int k_idx = m * ksize + n;
                sum += s_img[sm_idx] * kernel[k_idx];
            }
        }
        result[y * width + x] = sum;
    }
}

int main() {
    int width = 2880, height = 2800, ksize = 5;
    std::vector<int> imgFlat(width * height, 1);      // 測試用影像 (一維)
    std::vector<int> kernelFlat(ksize * ksize, 1);    // 5x5 kernel (一維)
    std::vector<int> resultFlat(width * height, 0);

    int *d_img, *d_kernel, *d_result;
    cudaMalloc(&d_img, imgFlat.size() * sizeof(int));
    cudaMalloc(&d_kernel, kernelFlat.size() * sizeof(int));
    cudaMalloc(&d_result, resultFlat.size() * sizeof(int));
    cudaMemcpy(d_img, imgFlat.data(), imgFlat.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernelFlat.data(), kernelFlat.size() * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int sharedMemSize = (BLOCK_SIZE + ksize - 1) * (BLOCK_SIZE + ksize - 1) * sizeof(int);

    conv2D_1Darray_cuda_shared<<<grid, block, sharedMemSize>>>(d_img, width, height, d_kernel, ksize, d_result);
    cudaMemcpy(resultFlat.data(), d_result, resultFlat.size() * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "卷積結果前10個像素：";
    for (int i = 0; i < 10; ++i) std::cout << resultFlat[i] << " ";
    std::cout << std::endl;

    cudaFree(d_img);
    cudaFree(d_kernel);
    cudaFree(d_result);
    return 0;
}
```
