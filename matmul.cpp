#include <iostream>
#include <chrono>
#include <vector>

// CPU Matrix Multiplication
void matrixMultCPU(
	const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int N
) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			float sum = 0.0f;
			for (int k = 0; k < N; k++) {
				sum += A[i * N + k] * B[k * N + j];
			}
			C[i * N + j] = sum;
		}
	}
}