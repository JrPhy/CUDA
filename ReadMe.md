CUDA (Compute Unified Device Architecture，統一計算架構) 是 NVIDIA 研發的平行運算平台及編程模型，可利用繪圖處理單元 (GPU) 的能力大幅提升運算效能。原本是拿來做繪圖運算的，其實就是矩陣運算，而現在神經網路也是利用矩陣計算，所以對於 GPU 需求特別高。一般程式所使用的記憶體是在 CPU 與 RAM 中，而 cuda 的變數就會多了在 GPU 與 GPU DRAM(GRAM) 中，所以就會需要將變數從 CPU 搬到 GPU，此種混合 CPU 與 GPU 的運算稱為異構運算。GPU 僅擅長加法與乘法運算，若有其他運算如除法或取餘數建議可以在 CPU 或是用其他方法取代

CPU 與 GPU 的設計理念不同，CPU 是設計來處理較複雜的事物，雖然也可跑 Multi Threads/Processes，但因為核心少所以不會有 GPU 快。GPU 的指令級沒有 CPU 多且較為簡單，做複雜指令需要呼叫許多指令，所以通常只做簡單的加法與乘法，但因為計算單元(Arithmetic logic unit, ALU)多，所以在做簡單的計算會非常快，若是每個計算獨立性高那就可以獲得指數成長，例如矩陣運算。
