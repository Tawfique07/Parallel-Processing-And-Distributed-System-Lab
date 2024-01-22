#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

const int M = 2;  // Rows of A
const int N = 3;  // Columns of A and Rows of B
const int P = 4;  // Columns of B
const int K = 3;  // Number of matrices

__global__ void matrixMultiplication(int *A, int *B, int *C, int m, int n, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < p) {
        int sum = 0;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * p + col];
        }
        C[row * p + col] = sum;
    }
}

void printMatrix(int *matrix, int rows, int cols, const char *name) {
    std::cout << name << ":\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void initializeRandomMatrix(int *matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i * cols + j] = rand() % 10;  // Adjust the range as needed
        }
    }
}

int main() {
    srand(time(NULL));  // Seed for random number generation

    int *h_A, *h_B, *h_C;  // Host matrices
    int *d_A, *d_B, *d_C;  // Device matrices

    // Allocate host memory
    h_A = (int *)malloc(K * M * N * sizeof(int));
    h_B = (int *)malloc(K * N * P * sizeof(int));
    h_C = (int *)malloc(K * M * P * sizeof(int));

    // Initialize host matrices A and B with random data
    for (int k = 0; k < K; ++k) {
        initializeRandomMatrix(h_A + k * M * N, M, N);
        initializeRandomMatrix(h_B + k * N * P, N, P);
    }

    // Print input matrices
    for (int k = 0; k < K; ++k) {
        printMatrix(h_A + k * M * N, M, N, ("Matrix A" + std::to_string(k)).c_str());
        printMatrix(h_B + k * N * P, N, P, ("Matrix B" + std::to_string(k)).c_str());
    }

    // Allocate device memory
    cudaMalloc((void **)&d_A, K * M * N * sizeof(int));
    cudaMalloc((void **)&d_B, K * N * P * sizeof(int));
    cudaMalloc((void **)&d_C, K * M * P * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, K * M * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * P * sizeof(int), cudaMemcpyHostToDevice);

    // Define thread block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((P + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);


    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Launch the matrix multiplication kernel for each pair of matrices
    for (int k = 0; k < K; ++k) {
        matrixMultiplication<<<gridDim, blockDim>>>(d_A + k * M * N, d_B + k * N * P, d_C + k * M * P, M, N, P);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milli = 0.0f;

    cudaEventElapsedTime(&milli, start, stop);

    // Copy the result back to the host
    cudaMemcpy(h_C, d_C, K * M * P * sizeof(int), cudaMemcpyDeviceToHost);

    // Print output matrices
    for (int k = 0; k < K; ++k) {
        printMatrix(h_C + k * M * P, M, P, ("Result Matrix" + std::to_string(k)).c_str());
    }

    cout << "time taken " <<milli;

    // Free device and host memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
