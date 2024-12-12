#include <iostream>
#include <chrono>
#include <fstream>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "matmul.h"

struct GPUMetrics {
    float kernel_time;
    float total_time;
    float h2d_time;
    float d2h_time;
    float gflops;
};

struct CPUMetrics {
    float time;
    float gflops;
};

CPUMetrics benchmarkCPU(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int N) {
    CPUMetrics metrics;
    
    auto start = std::chrono::high_resolution_clock::now();
    matrixMultCPU(A, B, C, N);
    auto end = std::chrono::high_resolution_clock::now();
    
    metrics.time = std::chrono::duration<float, std::milli>(end - start).count();
    float ops = 2.0f * N * N * N;  // multiply-add for each element
    metrics.gflops = (ops / (metrics.time * 1e6));
    
    return metrics;
}

GPUMetrics benchmarkGPU(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int N) {
    GPUMetrics metrics;
    
    // Allocate GPU memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));
    
    // Create CUDA events for timing
    cudaEvent_t start_h2d, stop_h2d, start_kernel, stop_kernel, start_d2h, stop_d2h;
    cudaEventCreate(&start_h2d);
    cudaEventCreate(&stop_h2d);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);
    cudaEventCreate(&start_d2h);
    cudaEventCreate(&stop_d2h);
    
    // Time H2D transfer
    cudaEventRecord(start_h2d);
    cudaMemcpy(d_A, A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(stop_h2d);
    cudaEventSynchronize(stop_h2d);
    cudaEventElapsedTime(&metrics.h2d_time, start_h2d, stop_h2d);
    
    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
    
    cudaEventRecord(start_kernel);
    matrixMultGPU<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop_kernel);
    cudaEventSynchronize(stop_kernel);
    cudaEventElapsedTime(&metrics.kernel_time, start_kernel, stop_kernel);
    
    // Time D2H transfer
    cudaEventRecord(start_d2h);
    cudaMemcpy(C.data(), d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop_d2h);
    cudaEventSynchronize(stop_d2h);
    cudaEventElapsedTime(&metrics.d2h_time, start_d2h, stop_d2h);
    
    // Calculate total time and GFLOPS
    metrics.total_time = metrics.h2d_time + metrics.kernel_time + metrics.d2h_time;
    float ops = 2.0f * N * N * N;
    metrics.gflops = (ops / (metrics.kernel_time * 1e6));
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return metrics;
}

void saveResults(const std::string& filename, int N, const CPUMetrics& cpu_metrics, const GPUMetrics& gpu_metrics, float max_diff) {
    std::ofstream outFile;
    // Open file in append mode
    outFile.open(filename, std::ios::app);
    
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    
    // Write header if file is empty
    outFile.seekp(0, std::ios::end);
    if (outFile.tellp() == 0) {
        outFile << "Matrix_Size,"
                << "CPU_Time_ms,CPU_GFLOPS,"
                << "GPU_Kernel_Time_ms,GPU_H2D_Time_ms,GPU_D2H_Time_ms,"
                << "GPU_Total_Time_ms,GPU_GFLOPS,Max_Difference\n";
    }
    
    // Write results in CSV format
    outFile << N << ","
            << cpu_metrics.time << "," << cpu_metrics.gflops << ","
            << gpu_metrics.kernel_time << "," << gpu_metrics.h2d_time << ","
            << gpu_metrics.d2h_time << "," << gpu_metrics.total_time << ","
            << gpu_metrics.gflops << "," << max_diff << "\n";
    
    outFile.close();
}

int main() {
    std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096, 8192};
    const std::string results_file = "results.csv";
    
    for (int N : sizes) {
        // Initialize matrices
        std::vector<float> A(N * N);
        std::vector<float> B(N * N);
        std::vector<float> C_cpu(N * N);
        std::vector<float> C_gpu(N * N);
        
        // Fill with random data
        unsigned int seed = 42;
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        
        for (int i = 0; i < N * N; i++) {
            A[i] = dis(gen);
            B[i] = dis(gen);
        }
        
        std::cout << "\nMatrix size: " << N << " x " << N << "\n";
        std::cout << "====================================\n";
        
        // CPU Benchmark
        auto cpu_metrics = benchmarkCPU(A, B, C_cpu, N);
        std::cout << "CPU Results:\n";
        std::cout << "  Time: " << cpu_metrics.time << " ms\n";
        std::cout << "  GFLOPS: " << cpu_metrics.gflops << "\n\n";
        
        // GPU Benchmark
        auto gpu_metrics = benchmarkGPU(A, B, C_gpu, N);
        std::cout << "GPU Results:\n";
        std::cout << "  Kernel Time: " << gpu_metrics.kernel_time << " ms\n";
        std::cout << "  H2D Transfer: " << gpu_metrics.h2d_time << " ms\n";
        std::cout << "  D2H Transfer: " << gpu_metrics.d2h_time << " ms\n";
        std::cout << "  Total Time: " << gpu_metrics.total_time << " ms\n";
        std::cout << "  GFLOPS: " << gpu_metrics.gflops << "\n";
        
        // Verify results
        float max_diff = 0.0f;
        for (int i = 0; i < N * N; i++) {
            max_diff = std::max(max_diff, std::abs(C_cpu[i] - C_gpu[i]));
        }
        std::cout << "  Max difference between CPU and GPU: " << max_diff << "\n";
        
        // Save results to file
        saveResults(results_file, N, cpu_metrics, gpu_metrics, max_diff);
    }
    
    std::cout << "\nResults have been saved to: " << results_file << std::endl;
    
    return 0;
}