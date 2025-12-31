
#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void conv1d_shared(const float* img, int img_size, const float* kernel, int kernel_size, float* out) {
    extern __shared__ float s_img[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    int k_radius = kernel_size / 2;

    // 載入資料到 shared memory
    if (i < img_size)
        s_img[tid] = img[i];
    else
        s_img[tid] = 0.0f;
    __syncthreads();

    // 卷積運算
    float sum = 0.0f;
    for (int k = 0; k < kernel_size; ++k) {
        int idx = tid + k - k_radius;
        int global_idx = i + k - k_radius;
        float val = 0.0f;
        // 先嘗試從 shared memory 取值
        if (idx >= 0 && idx < blockDim.x && global_idx >= 0 && global_idx < img_size)
            val = s_img[idx];
        // 若超出 block 範圍則直接從 global memory 取值
        else if (global_idx >= 0 && global_idx < img_size)
            val = img[global_idx];
        sum += val * kernel[k];
    }
    if (i < img_size)
        out[i] = sum;
}

int main() {
    const int img_size = 8;
    const int kernel_size = 3;
    float img[img_size] = {1, 2, 3, 4, 5, 6, 7, 8};
    float kernel[kernel_size] = {0.25, 0.5, 0.25};
    float out_gpu[img_size] = {0};

    float *d_img, *d_kernel, *d_out;
    cudaMalloc((void**)&d_img, img_size * sizeof(float));
    cudaMalloc((void**)&d_kernel, kernel_size * sizeof(float));
    cudaMalloc((void**)&d_out, img_size * sizeof(float));
    cudaMemcpy(d_img, img, img_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = BLOCK_SIZE;
    int gridSize = (img_size + blockSize - 1) / blockSize;
    int sharedMemSize = blockSize * sizeof(float);
    conv1d_shared<<<gridSize, blockSize, sharedMemSize>>>(d_img, img_size, d_kernel, kernel_size, d_out);
    cudaMemcpy(out_gpu, d_out, img_size * sizeof(float), cudaMemcpyDeviceToHost);

    printf("CUDA Shared Memory 結果:\n");
    for (int i = 0; i < img_size; ++i) printf("%.2f ", out_gpu[i]);
    printf("\n");

    cudaFree(d_img);
    cudaFree(d_kernel);
    cudaFree(d_out);

    return 0;
}
