在平行計算需要注意競爭條件(race condition)，cuda 中也提供了一些關於同步的函數，也就是會等待某個範圍內都計算好後在到下一條指令

1. cudaMemcpy: 在 CPU 與 GPU 間複製記憶體的值(隱式)
2. cudaDeviceSynchronize: 等待整个 GPU 完成所有任務
3. __syncthreads：保證同一個 block 內線程共享記憶體的一致性
4. cudaStreamSynchronize: 等待某個 stream 的所有操作完成。
5. cudaEventSynchronize: 等待某個事件完成，常用於跨 stream 同步。

前面兩個在一般情況下比較常見，後面兩個則是在**非同步操作**才會用到。在做計算時沒有等到全部同步完就去操作，那可能會得到非預期的結果，所以通常在呼叫 KERNEL 時後面都會搭配 cudaDeviceSynchronize();
