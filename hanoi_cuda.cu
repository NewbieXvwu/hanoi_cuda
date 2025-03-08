#include <windows.h>
#include <stdio.h>
#include <algorithm>
#include <string.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d code=%d (%s)\n", \
        __FILE__, __LINE__, err, cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// 修改关键点：使用更明确的常量内存声明方式
__device__ __constant__ int cuda_n;
__device__ __constant__ int cuda_direction;

__global__ void hanoi_kernel(int *d_steps, long long base, long long chunk_steps) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= chunk_steps) return;

    long long m = base + idx + 1;
    
    // 使用内建函数 __ffsll 查找最低位 1 的位置
    int disk = __ffsll(m) - 1;
    
    // 优化：避免使用模运算，改用位运算判断奇偶性
    int from = (disk & 1) ? 1 : 0;
    
    // 从常量内存读取值并预计算奇偶性
    int n_mod2 = cuda_n & 1;
    int direction = cuda_direction;
    
    // 优化：使用位运算获取移位后的特定位
    int shift_bit = (m >> (disk + 1)) & 1;
    int offset = shift_bit ? 1 : 2;
    
    // 使用预计算的 direction 替代实时计算
    int to = (from + direction * offset) % 3;
    from %= 3;
    
    // 使用预计算的 n_mod2
    from = (from + n_mod2) % 3;
    to = (to + n_mod2) % 3;
    
    d_steps[idx] = (from << 4) | to;
}

// 进度条生成辅助函数
const char* generate_progress_bar(long long current, long long total) {
    static char bar[21];
    memset(bar, ' ', 20);
    bar[20] = '\0';
    
    int progress = (int)(20.0 * current / total);
    for (int i = 0; i < progress && i < 20; ++i) {
        bar[i] = '=';
    }
    if (progress < 20) {
        bar[progress] = '>';
    }
    return bar;
}

double get_time() {
    LARGE_INTEGER freq, time;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&time);
    return (double)time.QuadPart / freq.QuadPart;
}

void solve_hanoi(int n) {
    long long total_steps = (1LL << n) - 1;
    const double start_time = get_time(); // 记录总体开始时间
    
    // 提前声明所有可能被跳过的变量
    float milliseconds = 0;  // 初始化为默认值
    double elapsed_sec = 0;
    double est_total = 0;
    double remaining_sec = 0;

    // 预计算常量并分别拷贝到常量内存
    int host_n = n;
    int host_direction = (n % 2) ? -1 : 1;
    CHECK_CUDA(cudaMemcpyToSymbol(cuda_n, &host_n, sizeof(int)));
    CHECK_CUDA(cudaMemcpyToSymbol(cuda_direction, &host_direction, sizeof(int)));

    // 复用显存指针（保留 static 优化）
    static int *d_steps = nullptr;
    static size_t allocated_size = 0;

    // 获取 GPU 设备属性以优化线程配置
    int deviceId;
    cudaDeviceProp props;
    CHECK_CUDA(cudaGetDevice(&deviceId));
    CHECK_CUDA(cudaGetDeviceProperties(&props, deviceId));
    
    // 根据计算能力选择最佳线程数
    int threads;
    if (props.major >= 9) {          // Hopper 及更新架构
        threads = 512;
    } else if (props.major >= 8) {   // Ampere 架构
        threads = 384;
    } else if (props.major >= 7) {   // Volta/Turing 架构
        threads = 384;
    } else {                         // Pascal 及更早架构
        threads = 256;
    }

    // 使用异步流进行内存操作
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    CHECK_CUDA(cudaEventRecord(start));
    
    // 初始化颜色支持（Windows 需要启用 VT 支持）
    #ifdef _WIN32
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode = 0;
    GetConsoleMode(hConsole, &mode);
    SetConsoleMode(hConsole, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
    #endif

    for (long long base = 0; base < total_steps; ) {
        size_t free_mem, total_mem;
        
        // 减少显存查询频率（每 10 次循环查询一次）
        if (base % 10 == 0) {
            CHECK_CUDA(cudaMemGetInfo(&free_mem, &total_mem));
        }
        
        // 使用更激进的内存分配策略（保留 512MB 安全边界）
        const size_t safety_margin = 512LL << 20;
        size_t available_mem = (free_mem > safety_margin) ? (free_mem - safety_margin) : 0;
        
        if (available_mem < sizeof(int)) {
            fprintf(stderr, "Insufficient GPU memory: Free %.2fGB < 512MB required\n", 
                   free_mem / 1024.0 / 1024 / 1024);
            goto cleanup;
        }
        
        // 计算最优块大小（使用动态调整的线程数）
        long long max_chunk_steps = available_mem / sizeof(int);
        long long remaining_steps = total_steps - base;
        long long chunk_steps = std::min(max_chunk_steps, remaining_steps);
        
        // 动态调整块大小（最小 1M，最大 1G）
        chunk_steps = std::min(chunk_steps, 1LL << 30);  // 上限 1G
        chunk_steps = std::max(chunk_steps, 1LL << 20);  // 下限 1M

        // 自适应更新频率
        long long update_interval = std::max(total_steps / 1000, 1LL << 20);
        chunk_steps = std::min(chunk_steps, update_interval);

        // 复用显存（仅当需要更大内存时才重新分配）
        if (chunk_steps > allocated_size) {
            if (d_steps) CHECK_CUDA(cudaFree(d_steps));
            CHECK_CUDA(cudaMallocAsync(&d_steps, chunk_steps * sizeof(int), stream));
            allocated_size = chunk_steps;
        }
        
        // 使用异步内核启动，应用优化的线程数
        dim3 blocks((chunk_steps + threads - 1) / threads);
        hanoi_kernel<<<blocks, threads, 0, stream>>>(d_steps, base, chunk_steps);
        
        // 进度显示逻辑
        elapsed_sec = (get_time() - start_time);
        est_total = (elapsed_sec * total_steps) / (base + 1);
        remaining_sec = est_total - elapsed_sec;
        
        printf("\r\x1b[36mProgress:\x1b[0m [\x1b[32m%-20s\x1b[0m] \x1b[33m%6.2f%%\x1b[0m | "
              "Chunk: \x1b[35m%5.2fGB\x1b[0m | "
              "Remaining: \x1b[31m%6.2fGB\x1b[0m | "
              "ETA: \x1b[34m%.1f sec\x1b[0m   ",
              generate_progress_bar(base, total_steps),
              base * 100.0 / total_steps,
              chunk_steps * sizeof(int) / (1024.0 * 1024 * 1024),
              (total_steps - base) * sizeof(int) / (1024.0 * 1024 * 1024),
              remaining_sec);
        fflush(stdout);

        CHECK_CUDA(cudaStreamSynchronize(stream));
        base += chunk_steps;
    }
    
    // 完成时换行
    printf("\n");
    
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

cleanup:
    // 打印纯 GPU 计算时间
    if (milliseconds > 0) {
        printf("\x1b[32mPure GPU compute time: %.2f ms\x1b[0m\n", milliseconds);
    }
    
    // 显存释放策略：小规模计算立即释放，大规模计算保留复用
    if (d_steps && n < 30) {  // 30 层以下立即释放
        CHECK_CUDA(cudaFree(d_steps));
        d_steps = nullptr;
        allocated_size = 0;
    }
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaStreamDestroy(stream));
}

int main() {
    int n;
    printf("Enter number of Hanoi layers: ");
    scanf("%d", &n);

    if (n < 1) {
        printf("Number of layers must be at least 1\n");
        return 1;
    }

    double start = get_time();
    solve_hanoi(n);
    double end = get_time();

    printf("Total time: %.2f seconds\n", end - start);

    printf("\nPress any key to exit...");
    getchar();
    getchar();

    return 0;
}
