#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>

#define TILE_SIZE 32
#define THREAD_TILE_2 2
#define THREAD_TILE_4 4

// checking for any errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = (call); \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(error) << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (false);

/**
 * Naive matrix multiplication
 * Each thread will compute one element of C
 * Repeated duplicate global reads
 * A = m x k
 * B = k x n
 * C = m x n
 */
__global__ void naiveMatrixMultiplication(float *A, float *B, float *C, int m, int n, int k) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < n && row < m) {
        float sum = 0.0f;

        for (int i = 0; i < k; i++) {
            sum += A[row * k + i] * B[i * n + col];
        }

        C[row * n + col] = sum;
    }
}

/**
 * Standard shared tile matrix multiplication
 * Tile size is the same as the block size
 * Each Thread in the block will load one value into shared tile at each iteration and calculate one value of C
 * A = m x k
 * B = k x n
 * C = m x n
 */
__global__ void sharedTileMatrixMultiplication(float *A, float *B, float *C, int m, int n, int k) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // global index of C
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    int numOfTiles = (k + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < numOfTiles; tile++) {
        int aCol = tile * TILE_SIZE + tx;
        int bRow = tile * TILE_SIZE + ty;

        // each thread loads one A value into shared memory
        if (row < m && aCol < k) {
            tileA[ty][tx] = A[row * k + aCol];
        } else {
            tileA[ty][tx] = 0.0f;
        }

        if (bRow < k && col < n) {
            tileB[ty][tx] = B[bRow * n + col];
        } else {
            tileB[ty][tx] = 0.0f;
        }

        // wait until every thread in the block has loaded data in the shared memory
        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++) {
            sum += tileA[ty][i] * tileB[i][tx];
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        C[row * n + col] = sum;
    }
}

/**
 * 1 x 2 register tilling i.e., each thread will calculate 2 adjacent elements in C
 * 
 * A = m x k
 * B = k x n
 * C = m x n
 */
__global__ void registerTilingMatrixMultiplication12(float *A, float *B, float *C, int m, int n, int k) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int localRow = ty;
    int localCol = THREAD_TILE_2 * tx;

    // the elements a single thread will calculate are C[row][col0] and C[row][col1]
    int row = blockIdx.y * TILE_SIZE + localRow;
    int col0 = blockIdx.x * TILE_SIZE + localCol;
    int col1 = col0 + 1;

    // final values that will be stored in C. 
    float sum0 = 0.0f;
    float sum1 = 0.0f;

    int numOfTiles = (k + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < numOfTiles; tile++) {
        // since each shared memory tile is 16 x 16 and each thread is handle 1x2 tile. The number of threads in each block will be 16 * 8
        // so each thread will load 2 values from A and 2 values from B

        int aCol0 = tile * TILE_SIZE + localCol;
        int aCol1 = aCol0 + 1;

        // load values from A
        tileA[localRow][localCol] = (row < m && aCol0 < k) ? A[row * k + aCol0] : 0.0f;
        tileA[localRow][localCol + 1] = (row < m && aCol1 < k) ? A[row * k + aCol1] : 0.0f;

        int bRow = tile * TILE_SIZE + localRow;

        // load values from B
        tileB[localRow][localCol] = (bRow < k && col0 < n) ? B[bRow * n + col0] : 0.0f;
        tileB[localRow][localCol + 1] = (bRow < k && col1 < n) ? B[bRow * n + col1] : 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++) {
            float a = tileA[localRow][i];

            float b0 = tileB[i][localCol];
            float b1 = tileB[i][localCol + 1];

            sum0 += a * b0;
            sum1 += a * b1;
        }

        __syncthreads();
    }

    if (row < m && col0 < n) {
        C[row * n + col0] = sum0;
    }
    
    if (row < m && col1 < n) {
        C[row * n + col1] = sum1;
    }
}

/**
 * 2 x 2 register tiling i.e., each thread will calculate a 2 x 2 tile in C
 * Goal is to reduce the number of shared memory accesses
 * A = m x k
 * B = k x n
 * C = m x n
 * 
 * Having large register tiling does not automatically make the kernel faster. Since more register requirement can lead to less occupancy
 */
__global__ void registerTilingMatrixMultiplication22(float *A, float *B, float *C, int m, int n, int k) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x; // local column
    int ty = threadIdx.y; // local row

    // since each thread is handling THREAD_TILE number of rows and cols.
    // these are the positions inside the tile
    int localRow = THREAD_TILE_2 * ty;
    int localCol = THREAD_TILE_2 * tx;

    // these four values together will form a 2x2 tile in C as C[row0][col0], C[row0][col1], C[row1][col0], C[row1][col1]
    int row0 = blockIdx.y * TILE_SIZE + localRow; // actual row of the element calculated in C
    int row1 = row0 + 1;

    int col0 = blockIdx.x * TILE_SIZE + localCol; // actual col of the element calculated in C
    int col1 = col0 + 1;

    // final values that will be stored in C. Stored in registers.
    float sum00 = 0.0f;
    float sum01 = 0.0f;
    float sum10 = 0.0f;
    float sum11 = 0.0f;
    
    int numOfTiles = (k + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < numOfTiles; tile++) {
        // since each tile is 16x16 but each thread is handling 4 elements. The number of threads in this case will be 8x8
        // so each thread will load 4 values from A and 4 values from B in the shared tile memory.

        // global cols from A
        int aCol0 = tile * TILE_SIZE + localCol;
        int aCol1 = aCol0 + 1;

        // loading values from A
        tileA[localRow][localCol] = (row0 < m && aCol0 < k) ? A[row0 * k + aCol0] : 0.0f;
        tileA[localRow][localCol + 1] = (row0 < m && aCol1 < k) ? A[row0 * k + aCol1] : 0.0f;
        tileA[localRow + 1][localCol] = (row1 < m && aCol0 < k) ? A[row1 * k + aCol0] : 0.0f;
        tileA[localRow + 1][localCol + 1] = (row1 < m && aCol1 < k) ? A[row1 * k + aCol1] : 0.0f;

        // global rows from B
        int bRow0 = tile * TILE_SIZE + localRow;
        int bRow1 = bRow0 + 1;

        // loading values from B
        tileB[localRow][localCol] = (bRow0 < k && col0 < n) ? B[bRow0 * n + col0] : 0.0f;
        tileB[localRow][localCol + 1] = (bRow0 < k && col1 < n) ? B[bRow0 * n + col1] : 0.0f;
        tileB[localRow + 1][localCol] = (bRow1 < k && col0 < n) ? B[bRow1 * n + col0] : 0.0f;
        tileB[localRow + 1][localCol + 1] = (bRow1 < k && col1 < n) ? B[bRow1 * n + col1] : 0.0f;

        // make sure that every thread has written to the shared memory
        __syncthreads();

        // compute the four output elements owned by this thread
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            float a0 = tileA[localRow][i];
            float a1 = tileA[localRow + 1][i];

            float b0 = tileB[i][localCol];
            float b1 = tileB[i][localCol + 1];

            sum00 += a0 * b0;
            sum01 += a0 * b1;
            sum10 += a1 * b0;
            sum11 += a1 * b1;
        }

        __syncthreads();
    }

    if (row0 < m && col0 < n) {
        C[row0 * n + col0] = sum00;
    }

    if (row0 < m && col1 < n) {
        C[row0 * n + col1] = sum01;
    }

    if (row1 < m && col0 < n) {
        C[row1 * n + col0] = sum10;
    }

    if (row1 < m && col1 < n) {
        C[row1 * n + col1] = sum11;
    }
}

/**
 * 4x4 register tiling
 * A = m x k
 * B = k x n
 * C = m x n
 * 
 * Just need to change the value of THREAD_TILE_4 to some other value for different register tiling size.
 */
__global__ void registerTilingMatrixMultiplication44(float *A, float *B, float *C, int m, int n, int k) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int localRow = THREAD_TILE_4 * ty;
    int localCol = THREAD_TILE_4 * tx;

    // these rows and cols values will together form the 4x4 tile in C
    int row0 = blockIdx.y * TILE_SIZE + localRow;
    int rows[THREAD_TILE_4];

    #pragma unroll
    for (int i = 0; i < THREAD_TILE_4; i++) {
        rows[i] = row0 + i;
    }

    int col0 = blockIdx.x * TILE_SIZE + localCol;
    int cols[THREAD_TILE_4];

    #pragma unroll
    for (int i = 0; i < THREAD_TILE_4; i++) {
        cols[i] = col0 + i;
    }

    float sums[THREAD_TILE_4][THREAD_TILE_4];

    #pragma unroll
    for (int i = 0; i < THREAD_TILE_4; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_4; j++) {
            sums[i][j] = 0.0f;
        }
    }

    int numOfTiles = (k + TILE_SIZE - 1)/TILE_SIZE;

    for (int tile = 0; tile < numOfTiles; tile++) {
        int aCol0 = tile * TILE_SIZE + localCol;

        // load values from A
        #pragma unroll
        for (int i = 0; i < THREAD_TILE_4; i++) {
            #pragma unroll
            for (int j = 0; j < THREAD_TILE_4; j++) {
                tileA[localRow + i][localCol + j] = rows[i] < m && (aCol0 + j) < k ? A[rows[i] * k + aCol0 + j] : 0.0f;
            }
        }

        int bRow0 = tile * TILE_SIZE + localRow;
        // load values from B
        #pragma unroll
        for (int i = 0; i < THREAD_TILE_4; i++) {
            #pragma unroll
            for (int j = 0; j < THREAD_TILE_4; j++) {
                tileB[localRow + i][localCol + j] = (bRow0 + i) < k && cols[j] < n ? B[(bRow0 + i) * n + cols[j]] : 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int h = 0; h < THREAD_TILE_4; h++) {
            #pragma unroll
            for (int i = 0; i < THREAD_TILE_4; i++) {
                for (int j = 0; j < TILE_SIZE; j++) {
                    sums[h][i] += tileA[localRow + h][j] * tileB[j][localCol + i];
                }
            }
        }

        __syncthreads();
    }

    for (int j = 0; j < TILE_SIZE; j++) {
        float regA[4];
        float regB[4];

        #pragma unroll
        for (int h = 0; h < 4; h++)
            regA[h] = tileA[localRow + h][j];

        #pragma unroll
        for (int i = 0; i < 4; i++)
            regB[i] = tileB[j][localCol + i];

        #pragma unroll
        for (int h = 0; h < 4; h++) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                sums[h][i] += regA[h] * regB[i];
            }
        }
    }
}

/**
 * Calculates the average kernel time over `iterations` number of iterations
 */
template <typename LaunchKernel>
float measureKernelTime(
    const char* kernelName,
    int iterations,
    LaunchKernel launchKernel
) {
    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up launch to remove first-launch initialization overhead.
    launchKernel();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < iterations; ++i) {
        launchKernel();
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float totalMilliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalMilliseconds, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float averageMilliseconds = totalMilliseconds / static_cast<float>(iterations);

    std::cout<< kernelName<< ": "<< averageMilliseconds<< " ms average over "<< iterations<< " iterations\n";

    return averageMilliseconds;
}

/**
 * Naive CPU matrix multiplication
 * This will be used as a reference to check correctness of kernel outputs
 */
void matrixMultiplyCPU(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int M,
    int K,
    int N
) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            double sum = 0.0;

            for (int i = 0; i < K; ++i) {
                sum += static_cast<double>(A[row * K + i]) *
                       static_cast<double>(B[i * N + col]);
            }

            C[row * N + col] = static_cast<float>(sum);
        }
    }
}

/**
 * Validate `actual` against `expected` element-wise.
 * Passes if absolute error <= `absoluteTolerance` or relative error <= `relativeTolerance`.
 * Prints the first failing element (with details) or a summary on success.
 */
bool checkCorrectness(
    const std::vector<float>& expected,
    const std::vector<float>& actual,
    const char* kernelName,
    float absoluteTolerance = 1e-2f,
    float relativeTolerance = 1e-3f
) {
    if (expected.size() != actual.size()) {
        std::cerr << kernelName << ": output sizes do not match\n";
        return false;
    }

    float maximumAbsoluteError = 0.0f;
    float maximumRelativeError = 0.0f;
    size_t maximumErrorIndex = 0;

    for (size_t i = 0; i < expected.size(); ++i) {
        float absoluteError = std::fabs(expected[i] - actual[i]);

        float denominator = std::max(std::fabs(expected[i]), 1e-6f);

        float relativeError = absoluteError / denominator;

        if (absoluteError > maximumAbsoluteError) {
            maximumAbsoluteError = absoluteError;
            maximumRelativeError = relativeError;
            maximumErrorIndex = i;
        }

        bool valueMatches = absoluteError <= absoluteTolerance || relativeError <= relativeTolerance;

        if (!valueMatches) {
            std::cerr
                << kernelName
                << ": FAILED at index "
                << i
                << "\nExpected: "
                << expected[i]
                << "\nActual:   "
                << actual[i]
                << "\nAbsolute error: "
                << absoluteError
                << "\nRelative error: "
                << relativeError
                << '\n';

            return false;
        }
    }

    std::cout
        << kernelName
        << ": PASSED"
        << " | max absolute error = "
        << maximumAbsoluteError
        << " at index "
        << maximumErrorIndex
        << " | relative error = "
        << maximumRelativeError
        << '\n';

    return true;
}

int main() {
    // A = M x K
    // B = K x N
    // C = M x N

    constexpr int M = 1 << 10;
    constexpr int K = 1 << 10;
    constexpr int N = 1 << 10;

    constexpr int ITERATIONS = 10;

    const size_t elementsA = static_cast<size_t>(M) * K;
    const size_t elementsB = static_cast<size_t>(K) * N;
    const size_t elementsC = static_cast<size_t>(M) * N;

    const size_t bytesA = elementsA * sizeof(float);
    const size_t bytesB = elementsB * sizeof(float);
    const size_t bytesC = elementsC * sizeof(float);

    std::vector<float> h_A(elementsA);
    std::vector<float> h_B(elementsB);

    std::vector<float> h_reference(elementsC);
    std::vector<float> h_naive(elementsC);
    std::vector<float> h_shared(elementsC);
    std::vector<float> h_register12(elementsC);
    std::vector<float> h_register22(elementsC);
    std::vector<float> h_register44(elementsC);

    // initializing the matrices
    for (size_t i = 0; i < h_A.size(); ++i) {
        h_A[i] = static_cast<float>((i % 100) + 1) / 100.0f;
    }

    for (size_t i = 0; i < h_B.size(); ++i) {
        h_B[i] = static_cast<float>((i % 50) + 1) / 50.0f;
    }

    std::cout << "Calculating CPU reference...\n";

    matrixMultiplyCPU(h_A, h_B, h_reference, M, K, N);

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytesA, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytesB, cudaMemcpyHostToDevice));

    dim3 standardBlock(TILE_SIZE, TILE_SIZE);

    dim3 standardGrid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    float naiveTime = measureKernelTime(
        "Naive kernel",
        ITERATIONS,
        [&]() {
            naiveMatrixMultiplication<<<standardGrid, standardBlock>>>(d_A, d_B, d_C, M, N, K);
        }
    );

    CUDA_CHECK(cudaMemcpy(h_naive.data(), d_C, bytesC, cudaMemcpyDeviceToHost));

    float sharedTime = measureKernelTime(
        "Shared-memory tiled kernel",
        ITERATIONS,
        [&]() {
            sharedTileMatrixMultiplication<<<standardGrid, standardBlock>>>(d_A, d_B, d_C, M, N, K);
        }
    );

    CUDA_CHECK(cudaMemcpy(h_shared.data(), d_C, bytesC, cudaMemcpyDeviceToHost));

    dim3 register12Block(TILE_SIZE / THREAD_TILE_2, TILE_SIZE);

    dim3 register12Grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    float register12Time = measureKernelTime(
        "Register-tiled 1x2 kernel",
        ITERATIONS,
        [&]() {
            registerTilingMatrixMultiplication12<<<register12Grid, register12Block>>>(d_A, d_B, d_C, M, N, K);
        }
    );

    CUDA_CHECK(cudaMemcpy(h_register12.data(), d_C, bytesC, cudaMemcpyDeviceToHost));

    dim3 register22Block(TILE_SIZE / THREAD_TILE_2, TILE_SIZE / THREAD_TILE_2);

    dim3 register22Grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    float register22Time = measureKernelTime(
        "Register-tiled 2x2 kernel",
        ITERATIONS,
        [&]() {
            registerTilingMatrixMultiplication22<<<register22Grid, register22Block>>>(d_A, d_B, d_C, M, N, K);
        }
    );

    CUDA_CHECK(cudaMemcpy(h_register22.data(), d_C, bytesC, cudaMemcpyDeviceToHost));

    dim3 register44Block(TILE_SIZE / THREAD_TILE_4, TILE_SIZE / THREAD_TILE_4);

    dim3 register44Grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    float register44Time = measureKernelTime(
        "Register-tiled 4x4 Kernel",
        ITERATIONS,
        [&]() {
            registerTilingMatrixMultiplication44<<<register44Grid, register44Block>>>(d_A, d_B, d_C, M, N, K);
        }
    );

    CUDA_CHECK(cudaMemcpy(h_register44.data(), d_C, bytesC, cudaMemcpyDeviceToHost));

    std::cout << "\nCorrectness checks:\n";

    const bool naiveCorrect = checkCorrectness(h_reference, h_naive, "Naive kernel");

    const bool sharedCorrect = checkCorrectness(h_reference, h_shared, "Shared-memory tiled kernel");

    const bool register12Correct = checkCorrectness(h_reference, h_register12, "Register-tiled 1x2 kernel");

    const bool register22Correct = checkCorrectness(h_reference, h_register22, "Register-tiled 2x2 kernel");

    const bool register44Correct = checkCorrectness(h_reference, h_register44, "Register-tiled 4x4 kernel");

    const bool allCorrect = naiveCorrect && sharedCorrect && register12Correct && register22Correct && register44Correct;

    std::cout << "\nSpeedups relative to naive:\n";

    std::cout << "Shared tiled: " << naiveTime / sharedTime << "x\n";

    std::cout << "Register tiled 1x2: " << naiveTime / register12Time << "x\n";

    std::cout << "Register tiled 2x2: " << naiveTime / register22Time << "x\n";

    std::cout << "Register tiled 4x4: " << naiveTime / register44Time << "x\n";

    std::cout << "\nOverall correctness: " << (allCorrect ? "PASSED" : "FAILED") << '\n';

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return allCorrect ? EXIT_SUCCESS : EXIT_FAILURE;
}