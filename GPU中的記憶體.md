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
在前面提到的[矩陣乘法](https://github.com/JrPhy/CUDA/blob/main/%E5%90%8C%E6%AD%A5syncronize.md#2-__syncthreads)就有用到 Shared Memory，當然也可以不用，但速度就比較慢，因為要每次去存取 Global Memory 的值。

