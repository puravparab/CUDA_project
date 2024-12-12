#include <iostream>
#include <chrono>
#include <vector>
#include <fstream>
#include <random>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "matmul.h"

struct PerformanceMetrics {
	float cpu_time;           // ms
	float gpu_kernel_time;    // ms
	float mem_h2d_time;       // Host to Device transfer time
	float mem_d2h_time;       // Device to Host transfer time
	float total_gpu_time;     // Total GPU time including transfers
	float memory_bandwidth;   // GB/s
	float gflops;            // Giga FLOPS
	float occupancy;         // Thread occupancy
	int matrix_size;         // N x N
};

void saveMetricsToCSV(const PerformanceMetrics& metrics);

void measurePerformance(
	int N, std::vector<float>& A, std::vector<float>& B, std::vector<float>& C_cpu, std::vector<float>& C_gpu
) {
	PerformanceMetrics metrics;
	metrics.matrix_size = N;
	
	// Calculate theoretical FLOPS
	// Matrix multiplication requires 2*N^3 floating point operations
	float theoretical_flops = 2.0f * N * N * N;
	
	// CPU Timing
	auto start_cpu = std::chrono::high_resolution_clock::now();
	matrixMultCPU(A, B, C_cpu, N);
	auto end_cpu = std::chrono::high_resolution_clock::now();
	metrics.cpu_time = std::chrono::duration<float, std::milli>(end_cpu - start_cpu).count();
	
	// GPU Setup
	float *d_A, *d_B, *d_C;
	cudaMalloc(&d_A, N * N * sizeof(float));
	cudaMalloc(&d_B, N * N * sizeof(float));
	cudaMalloc(&d_C, N * N * sizeof(float));
	
	// Memory transfer timing (H2D)
	cudaEvent_t start_h2d, stop_h2d;
	cudaEventCreate(&start_h2d);
	cudaEventCreate(&stop_h2d);
	
	cudaEventRecord(start_h2d);
	cudaMemcpy(d_A, A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaEventRecord(stop_h2d);
	cudaEventSynchronize(stop_h2d);
	cudaEventElapsedTime(&metrics.mem_h2d_time, start_h2d, stop_h2d);
	
	// Kernel execution timing
	dim3 blockDim(16, 16);
	dim3 gridDim((N + blockDim.x - 1) / blockDim.x, 
								(N + blockDim.y - 1) / blockDim.y);
	
	cudaEvent_t start_kernel, stop_kernel;
	cudaEventCreate(&start_kernel);
	cudaEventCreate(&stop_kernel);
	
	cudaEventRecord(start_kernel);
	matrixMultGPU<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
	cudaEventRecord(stop_kernel);
	cudaEventSynchronize(stop_kernel);
	cudaEventElapsedTime(&metrics.gpu_kernel_time, start_kernel, stop_kernel);
	
	// Memory transfer timing (D2H)
	cudaEvent_t start_d2h, stop_d2h;
	cudaEventCreate(&start_d2h);
	cudaEventCreate(&stop_d2h);
	
	cudaEventRecord(start_d2h);
	cudaMemcpy(C_gpu.data(), d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop_d2h);
	cudaEventSynchronize(stop_d2h);
	cudaEventElapsedTime(&metrics.mem_d2h_time, start_d2h, stop_d2h);
	
	// Calculate derived metrics
	metrics.total_gpu_time = metrics.mem_h2d_time + metrics.gpu_kernel_time + metrics.mem_d2h_time;
	metrics.gflops = (theoretical_flops / metrics.gpu_kernel_time) / 1e6; // Convert to GFLOPS
	
	// Calculate memory bandwidth
	size_t total_bytes = 3 * N * N * sizeof(float); // 2 reads + 1 write
	metrics.memory_bandwidth = (total_bytes / metrics.gpu_kernel_time) / 1e6; // GB/s
	
	// Get occupancy
	int max_blocks_per_sm;
	int block_size = blockDim.x * blockDim.y;
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(
			&max_blocks_per_sm, matrixMultGPU, block_size, 0);
	
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	metrics.occupancy = (float)(max_blocks_per_sm * block_size) / 
											prop.maxThreadsPerMultiProcessor;
	
	// Save metrics to CSV
	saveMetricsToCSV(metrics);
	
	// Cleanup
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

void saveMetricsToCSV(const PerformanceMetrics& metrics) {
	static bool first_write = true;
	std::ofstream outFile;
	if (first_write) {
		outFile.open("detailed_performance.csv");
		outFile << "MatrixSize,CPU_Time,GPU_Kernel_Time,Memory_H2D_Time,Memory_D2H_Time,"
						<< "Total_GPU_Time,Memory_Bandwidth,GFLOPS,Occupancy\n";
		first_write = false;
	} else {
		outFile.open("detailed_performance.csv", std::ios::app);
	}
	
	outFile << metrics.matrix_size << ","
					<< metrics.cpu_time << ","
					<< metrics.gpu_kernel_time << ","
					<< metrics.mem_h2d_time << ","
					<< metrics.mem_d2h_time << ","
					<< metrics.total_gpu_time << ","
					<< metrics.memory_bandwidth << ","
					<< metrics.gflops << ","
					<< metrics.occupancy << "\n";
}

int main() {
	// Set a fixed seed for reproducible results
	srand(42);  // You can choose any integer as the seed

	// Test different matrix sizes
	for (int N : {128, 256, 512, 1024}) {
		// Initialize matrices
		std::vector<float> A(N * N);
		std::vector<float> B(N * N);
		std::vector<float> C_cpu(N * N);
		std::vector<float> C_gpu(N * N);
		
		// Initialize with random values
		for (int i = 0; i < N * N; i++) {
			A[i] = rand() / (float)RAND_MAX;
			B[i] = rand() / (float)RAND_MAX;
		}

		measurePerformance(N, A, B, C_cpu, C_gpu);
	}
	return 0;
}