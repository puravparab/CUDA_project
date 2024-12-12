#ifndef MATMUL_H
#define MATMUL_H

#include <vector>

// CPU Matrix Multiplication declaration
void matrixMultCPU(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int N);

// GPU Matrix Multiplication declaration
__global__ void matrixMultGPU(const float* A, const float* B, float* C, int N);

#endif