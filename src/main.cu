#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <map>
#include <sys/time.h>
#include <valarray>

#include <hdf5.h>

#include "range.hpp"
#include "utils.hpp"

#define NUM_ROWS 28
#define NUM_COLS 28
#define NUM_CHANNELS 1
#define NUM_DIGITS 10

// Same as block dimension
#define TILE_WIDTH 16
#define MULT_TW 32

// Wild assumption
#define CUDA_MAX_NUM_THREADS 1024

static int FLAGS_batch_size = 10000;
static std::string FLAGS_testdata{};
static std::string FLAGS_model{};

// Data and reference data dimensions
static int xdims[] = {FLAGS_batch_size, NUM_ROWS, NUM_COLS, NUM_CHANNELS};
static int rdims[] = {FLAGS_batch_size, NUM_DIGITS};

// Model dimensions
static int conv1dims[] = {5, 5, 1, 32};
static int conv2dims[] = {5, 5, 32, 64};
static int fc1dims[]   = {1024, 128};
static int fc2dims[]   = {128, 10};

float * fullyForwardOut2;

// CUDA device buffers
float * deviceOutputConv1;  // conv 1 vars
float * deviceInputPool1, * deviceOutputPool1;  // pool 1 vars
float * deviceOutputConv2;  // conv 2 vars
float * deviceInputPool2, * deviceOutputPool2;  // pool 2 vars
float * deviceInputFullyForward1, * deviceMaskFullyForward1, * deviceOutputFullyForward1; // fully connected 1 vars
float * deviceInputFullyForward2, * deviceMaskFullyForward2, * deviceOutputFullyForward2; // fully connected 2 vars

// Buffers for conv layer 1
float * device_X0_L1, * device_X1_L1, * device_X2_L1;
float * device_X_unrolled0_L1, * device_X_unrolled1_L1, * device_X_unrolled2_L1;
float * device_W_L1, * device_W_unrolled_L1;
float * device_Y_unrolled_L1;

// Buffers for conv layer 2
float * device_X_L2;
float * device_X_unrolled0_L2, * device_X_unrolled1_L2, * device_X_unrolled2_L2;
float * device_W_L2, * device_W_unrolled_L2;
float * device_Y_unrolled_L2;


static int loadData(float *x, float *y) {
    // Open the data file
    const auto file_id =
        H5Fopen(FLAGS_testdata.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);

    // Open the dataset x and y
    const auto x_id = H5Dopen2(file_id, "/x", H5P_DEFAULT);
    const auto y_id = H5Dopen2(file_id, "/y", H5P_DEFAULT);

    // Get the dataset x dimensions
    const auto xspace = H5Dget_space(x_id);
    const auto xndims = H5Sget_simple_extent_ndims(xspace);
    assert(xndims == 4);

    hsize_t input_dims[xndims];
    H5Sget_simple_extent_dims(xspace, input_dims, NULL);
    if (input_dims[0] != FLAGS_batch_size) {
        std::cout << "data size does not match batch size specified!\n";
        return 1; // return error
    }
    std::cout << "input dimensions = " << input_dims[0] << " x " << input_dims[1]
              << " x " << input_dims[2] << " x " << input_dims[3] << "\n";

    // Read the dataset x and y
    check_success(
        H5Dread(x_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, x));
    check_success(
        H5Dread(y_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, y));

    // Close the dataset x and y
    check_success(H5Dclose(x_id));
    check_success(H5Dclose(y_id));

    // Close the file
    check_success(H5Fclose(file_id));

    // return success
    return 0;
}

static void loadModel(float *conv1, float *conv2, float *fc1, float *fc2) {
    // Open the model file
    const auto file_id = H5Fopen(FLAGS_model.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);

    // Open the dataset
    const auto conv1_id = H5Dopen2(file_id, "/conv1", H5P_DEFAULT);
    const auto conv2_id = H5Dopen2(file_id, "/conv2", H5P_DEFAULT);
    const auto fc1_id   = H5Dopen2(file_id, "/fc1", H5P_DEFAULT);
    const auto fc2_id   = H5Dopen2(file_id, "/fc2", H5P_DEFAULT);

    // Read the dataset
    check_success(H5Dread(conv1_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                          H5P_DEFAULT, conv1));
    check_success(H5Dread(conv2_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                          H5P_DEFAULT, conv2));
    check_success(
        H5Dread(fc1_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, fc1));
    check_success(
        H5Dread(fc2_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, fc2));

    // Close the dataset x and y
    check_success(H5Dclose(conv1_id));
    check_success(H5Dclose(conv2_id));
    check_success(H5Dclose(fc1_id));
    check_success(H5Dclose(fc2_id));

    // Close the file
    check_success(H5Fclose(file_id));
}

// Kernel to compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {

    // Declare shared memory tiles Ads and Bds
    __shared__ float Ads[MULT_TW][MULT_TW];
    __shared__ float Bds[MULT_TW][MULT_TW];

    // Get row and column of the output element this thread is working on
    int CRow = blockIdx.y*blockDim.y + threadIdx.y;
    int CCol = blockIdx.x*blockDim.x + threadIdx.x;

    // Idea: in each "phase," have threads collaboratively load subsets of A and B elements
    // into the shared memory before they individually use these elements in dot product calculation.
    // These A and B subsets are referred to as "tiles"; they are the same size as the block dimensions
    // (16 in our case).  We need enough phases such that the tiles span the entire image- we'll also
    // need checks to ensure our algorithm works in the case of output dimensions that are not a multiple of
    // the tile width.

    int phase; // loop variable for phases
    int dot; // loop variable for dot product in each phase
    float Cval = 0.0f; // Holds accumulating value of output element


      // Including a +1 accounts for the case in which dimensions are not a multiple of TILE_WIDTH
      // Why can't we use ceil??????
      for(phase = 0; phase < (numAColumns-1)/MULT_TW + 1; phase++)
      {
        // Don't try to load nonexistent elements
        if((CRow < numCRows) && ((threadIdx.x + phase*MULT_TW) < numAColumns))
        {
          // Each thread loads an element into shared memory
          Ads[threadIdx.y][threadIdx.x] = A[CRow*numAColumns + threadIdx.x + phase*MULT_TW];
        }

        // Don't try to load nonexistent elements
        // Note: A and B have to be checked separately as they could have wildly different dimensions
        if((CCol < numCColumns) && ((threadIdx.y + phase*MULT_TW) < numBRows))
        {
          // Each thread loads an element into shared memory
          Bds[threadIdx.y][threadIdx.x] = B[(threadIdx.y + phase*MULT_TW)*numBColumns + CCol];
        }

        __syncthreads(); // Necessary to ensure all threads have loaded their data before proceeding with computation

        for(dot = 0; dot < MULT_TW; dot++) // Perform the dot product operation for current tile
        {
          // Verify that the tile elements don't step outside the bounds of our actual input matrices.
          // Necessary when numAColumns % TILE_WIDTH != 0
          if(((dot + phase*MULT_TW) < numAColumns) && ((dot + phase*MULT_TW) < numBRows))
             Cval += Ads[threadIdx.y][dot]*Bds[dot][threadIdx.x];
        }

        __syncthreads(); // Necessary to ensure all threads have finished computation before overwriting shared memory.
      }

      // Check that the thread is mapped to a valid element
      // Note: this cannot be performed above because threads that point outside the bounds of the output
      // matrix are still needed to load tiles into shared memory.
      if((CRow < numCRows) && (CCol < numCColumns))
      {
        C[CRow*numCColumns + CCol] = Cval;
      }
}

// Parallel implementation of the filter unroll
// Effectively reorganizes the filter array into a different format
__global__ void unrollFilters(int C, int M, int K, float * W, float * W_unroll)
{
  int p, q, c, m;

  m = threadIdx.x;
  c = blockIdx.x;

  if(c < C && m < M)
  {
    for(p = 0; p < K; p++)
    {
      for(q = 0; q < K; q++)
      {
        W_unroll[m*C*K*K + c*K*K + p*K + q] = W[p*K*C*M + q*C*M + c*M + m]; // Should get coalesced read access!
      }
    }
  }
}

// Parallel implementation of the output reroll
// Effectively reorganizes the output maps into a different format
__global__ void rerollOutput(int M, int N, int H_out, int W_out, float * Y_unrolled, float * Y)
{
  int m, n, h, w;

  m = threadIdx.x;
  n = blockIdx.x;

  if(m < M && n < N)
  {
    for(h = 0; h < H_out; h++)
    {
      for(w = 0; w < W_out; w++)
      {
        Y[((n * H_out + h) * W_out + w) *M + m] = Y_unrolled[n*M*H_out*W_out + m*H_out*W_out + h*W_out + w];
      }
    }
  }
}

// Parallel input unroll implementation
// Modified to work with streams
__global__ void unroll_gpu(int C, int H, int W, int K, float* X, float* X_unroll)
{
  int c;
  int s;
  int h_out, w_out, h_unroll, w_unroll, h_base;
  int x_index;

  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int H_out = (H - K + 1);
  int W_out = (W - K + 1);
  int W_unroll = H_out * W_out;

  if (t < C * W_unroll)
  {
    c = t % C; // Idea: change thread-to-element mapping to get coalesced memory access
    s = t/C;
    h_out = s/W_out;
    w_out = s % W_out;
    w_unroll = h_out * W_out + w_out;
    h_base = c * K * K;
    for(int p = 0; p < K; p++)
    {
      for(int q = 0; q < K; q++)
      {
        h_unroll = h_base + p * K + q;
        x_index = (h_out+p)*W*C + (w_out+q)*C + c;
        X_unroll[h_unroll*W_unroll + w_unroll] = X[x_index]; // Read accesses should be at least partially coalesced now
      }
    }
  }
}

// Forward convolutional layer: uses unrolling + matrix multiplication!
void convLayer_forward_streamed(int N, int M, int C, int H, int W, int K, float* X, float* Mask, float* device_Y, const int ydims[4], float * device_X0, float * device_X1, float * device_X2, float * device_X_unrolled0, float * device_X_unrolled1, float * device_X_unrolled2,  float * device_W, float * device_W_unrolled, float * device_Y_unrolled)
{
  int W_out = W - K + 1;
  int H_out = H - K + 1;
  int H_unroll = C * K * K;
  int W_unroll = H_out * W_out;

  // dim3's for filter unrolling
  dim3 blockDimension1(M, 1, 1);
  dim3 gridDimension1(C, 1, 1);

  // Copy memory and launch kernel to unroll the filter
  check_success(cudaMemcpy(device_W, Mask, M*C*K*K * sizeof(float), cudaMemcpyHostToDevice));

  unrollFilters<<<gridDimension1, blockDimension1>>>(C, M, K, device_W, device_W_unrolled);
  cudaDeviceSynchronize();

  // Initialize the grid and block dimensions for unrolling
  dim3 blockDimensionU(CUDA_MAX_NUM_THREADS, 1, 1);
  dim3 gridDimensionU(ceil((1.0*C*H_out*W_out)/CUDA_MAX_NUM_THREADS), 1, 1);

  // Initialize the grid and block dimensions for matrix multiplication
  dim3 blockDimension2(MULT_TW, MULT_TW, 1);
  dim3 gridDimension2(ceil((1.0*W_unroll)/MULT_TW), ceil((1.0*M)/MULT_TW), 1);

  // Create CUDA streams
  cudaStream_t stream0, stream1, stream2;
  cudaStreamCreate(&stream0);
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  // Unroll input using multiple kernel launches with streams
  for (int n = 0; n < N; n += 3)
  {
    // Copy over input memory to the device_X
    check_success(cudaMemcpyAsync(device_X0, X+(n*C*H*W), C * H * W * sizeof(float), cudaMemcpyHostToDevice, stream0));
    if(n+1 < N){
      check_success(cudaMemcpyAsync(device_X1, X+((n+1)*C*H*W), C * H * W * sizeof(float), cudaMemcpyHostToDevice, stream1));
    }
    if(n+2 < N){
      check_success(cudaMemcpyAsync(device_X2, X+((n+2)*C*H*W), C * H * W * sizeof(float), cudaMemcpyHostToDevice, stream2));
    }

    // Parallel input unroll
    unroll_gpu<<<gridDimensionU, blockDimensionU, 0, stream0>>>(C, H, W, K, device_X0, device_X_unrolled0);
    if(n+1 < N){
      unroll_gpu<<<gridDimensionU, blockDimensionU, 0, stream1>>>(C, H, W, K, device_X1, device_X_unrolled1);
    }
    if(n+2 < N){
      unroll_gpu<<<gridDimensionU, blockDimensionU, 0, stream2>>>(C, H, W, K, device_X2, device_X_unrolled2);
    }
    // cudaDeviceSynchronize();

    // Matrix multiplication
    matrixMultiplyShared<<<gridDimension2, blockDimension2, 0 , stream0>>>(device_W_unrolled, device_X_unrolled0, &(device_Y_unrolled[n*ydims[1]*ydims[2]*ydims[3]]),
      M, C*K*K, H_unroll, W_unroll, M, W_unroll);
    if(n+1 < N){
      matrixMultiplyShared<<<gridDimension2, blockDimension2, 0 , stream1>>>(device_W_unrolled, device_X_unrolled1, &(device_Y_unrolled[(n+1)*ydims[1]*ydims[2]*ydims[3]]),
        M, C*K*K, H_unroll, W_unroll, M, W_unroll);
    }
    if(n+2 < N){
      matrixMultiplyShared<<<gridDimension2, blockDimension2, 0 , stream2>>>(device_W_unrolled, device_X_unrolled2, &(device_Y_unrolled[(n+2)*ydims[1]*ydims[2]*ydims[3]]),
        M, C*K*K, H_unroll, W_unroll, M, W_unroll);
    }

  }

  cudaDeviceSynchronize();

  // Now "re-roll" the output Y
  dim3 blockDimension3(M, 1, 1);
  dim3 gridDimension3(N, 1, 1);

  rerollOutput<<<gridDimension3, blockDimension3>>>(M, N, H_out, W_out, device_Y_unrolled, device_Y);
  cudaDeviceSynchronize();

  // free memory
  cudaFree(device_X_unrolled0);
  cudaFree(device_X_unrolled1);
  cudaFree(device_X_unrolled2);
  cudaFree(device_X0);
  cudaFree(device_X1);
  cudaFree(device_X2);
  cudaFree(device_W_unrolled);
  cudaFree(device_Y_unrolled);
  cudaFree(device_W);
}

// Forward convolutional layer: uses unrolling + matrix multiplication! but no streams
void convLayer_forward_reg(int N, int M, int C, int H, int W, int K, float* Mask, float* device_Y, const int ydims[4], float * device_X, float * device_X_unrolled0, float * device_X_unrolled1, float * device_X_unrolled2, float * device_W, float * device_W_unrolled, float * device_Y_unrolled)
{
  int W_out = W - K + 1;
  int H_out = H - K + 1;
  int H_unroll = C * K * K;
  int W_unroll = H_out * W_out;

  // dim3's for filter unrolling
  dim3 blockDimension1(M, 1, 1);
  dim3 gridDimension1(C, 1, 1);

  // Create CUDA streams
  cudaStream_t stream0, stream1, stream2;
  cudaStreamCreate(&stream0);
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  // Copy memory and launch kernel to unroll the filter
  check_success(cudaMemcpy(device_W, Mask, M*C*K*K * sizeof(float), cudaMemcpyHostToDevice));

  unrollFilters<<<gridDimension1, blockDimension1>>>(C, M, K, device_W, device_W_unrolled);
  cudaDeviceSynchronize();

  // Initialize the grid and block dimensions for unrolling
  dim3 blockDimensionU(CUDA_MAX_NUM_THREADS, 1, 1);
  dim3 gridDimensionU(ceil((1.0*C*H_out*W_out)/CUDA_MAX_NUM_THREADS), 1, 1);

  // Initialize the grid and block dimensions for matrix multiplication
  dim3 blockDimension2(MULT_TW, MULT_TW, 1);
  dim3 gridDimension2(ceil((1.0*W_unroll)/MULT_TW), ceil((1.0*M)/MULT_TW), 1);

  // Unroll input using multiple kernel launches with streams
  for (int n = 0; n < N; n+=3)
  {
    // Parallel input unroll
    unroll_gpu<<<gridDimensionU, blockDimensionU, 0, stream0>>>(C, H, W, K, device_X+n*C*H*W, device_X_unrolled0);
    if(n+1 < N){
      unroll_gpu<<<gridDimensionU, blockDimensionU, 0, stream1>>>(C, H, W, K, device_X+(n+1)*C*H*W, device_X_unrolled1);
    }
    if(n+2 < N){
      unroll_gpu<<<gridDimensionU, blockDimensionU, 0, stream2>>>(C, H, W, K, device_X+(n+2)*C*H*W, device_X_unrolled2);
    }
    //cudaStreamSynchronize(stream0);

    // Matrix multiplication
    matrixMultiplyShared<<<gridDimension2, blockDimension2, 0 , stream0>>>(device_W_unrolled, device_X_unrolled0, &(device_Y_unrolled[n*ydims[1]*ydims[2]*ydims[3]]),
      M, C*K*K, H_unroll, W_unroll, M, W_unroll);
    if(n+1 < N){
      //cudaStreamSynchronize(stream1);
      matrixMultiplyShared<<<gridDimension2, blockDimension2, 0 , stream1>>>(device_W_unrolled, device_X_unrolled1, &(device_Y_unrolled[(n+1)*ydims[1]*ydims[2]*ydims[3]]),
        M, C*K*K, H_unroll, W_unroll, M, W_unroll);
    }
    if(n+2 < N){
      //cudaStreamSynchronize(stream2);
      matrixMultiplyShared<<<gridDimension2, blockDimension2, 0 , stream2>>>(device_W_unrolled, device_X_unrolled2, &(device_Y_unrolled[(n+2)*ydims[1]*ydims[2]*ydims[3]]),
        M, C*K*K, H_unroll, W_unroll, M, W_unroll);
    }

  }

  cudaDeviceSynchronize();

  // Now "re-roll" the output Y
  dim3 blockDimension3(M, 1, 1);
  dim3 gridDimension3(N, 1, 1);

  rerollOutput<<<gridDimension3, blockDimension3>>>(M, N, H_out, W_out, device_Y_unrolled, device_Y);
  cudaDeviceSynchronize();

  // free memory
  cudaFree(device_X_unrolled0);
  cudaFree(device_X_unrolled1);
  cudaFree(device_X_unrolled2);
  cudaFree(device_X);
  cudaFree(device_W_unrolled);
  cudaFree(device_Y_unrolled);
  cudaFree(device_W);
}


// CUDA kernel for average pool
// Uses standard mapping: one thread per output element
__global__ void average_pool_kernel(float *X, float *Y, int xdims_1, int xdims_2, int xdims_3, int ydims_1, int ydims_2, int ydims_3, int pool_size, int n) {

    int m, h, w;
    m = blockIdx.y;

    h = (blockIdx.z / ((ydims_2-1)/TILE_WIDTH + 1))*TILE_WIDTH + threadIdx.y;
    w = (blockIdx.z % ((ydims_1-1)/TILE_WIDTH + 1))*TILE_WIDTH + threadIdx.x;

    if(w < ydims_2 && h < ydims_1){
        float acc = 0.0f;
        for (int p = 0; p < pool_size; p++) {
            for (int q = 0; q < pool_size; q++) {
                int xoffset = n * xdims_1 * xdims_2 * xdims_3 + (pool_size * h + p) * xdims_2 * xdims_3 + (pool_size * w + q) * xdims_3 + m;
                acc += X[xoffset] / (1.0f * pool_size * pool_size);
            }
        }
        int yoffset = ((n * ydims_1 + h) * ydims_2 + w) * ydims_3 + m;
        Y[yoffset] = acc;
    }
}

// Choose the guess with largest score
static void argmax(const float *X, const int xdims[2], int *Y) {
    for (const auto i : range(0, xdims[0])) {
        auto max_idx = 0;
        auto max     = X[i * xdims[1]];
        for (const auto j : range(0, xdims[1])) {
            const auto elem = X[(i * xdims[1]) + j];
            if (elem > max) {
                max_idx = j;
                max     = elem;
            }
        }
        Y[i] = max_idx;
    }
}

// RELU
__global__ void relu_gpu(float *X, const int bounds)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;

    if(i < bounds)
    {
      X[i] = (X[i] < 0) ? 0 : X[i];
    }
}

void average_pool_streamed(int N, int M, int Z, int pool_size, float * deviceInputPool, float * deviceOutputPool, const int xdims[4], const int ydims[4])
{
    // Create CUDA streams
    cudaStream_t stream0, stream1, stream2;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    dim3 blockDimPool(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDimPool(1, M, Z);


    // Unroll input using multiple kernel launches with streams
    for (int n = 0; n < N; n+=3)
    {
        // Parallel input unroll
        average_pool_kernel<<<gridDimPool, blockDimPool, 0, stream0>>>(deviceInputPool, deviceOutputPool, xdims[1], xdims[2], xdims[3], ydims[1], ydims[2], ydims[3], pool_size, n);
        if(n+1 < N){
            average_pool_kernel<<<gridDimPool, blockDimPool, 0, stream1>>>(deviceInputPool, deviceOutputPool, xdims[1], xdims[2], xdims[3], ydims[1], ydims[2], ydims[3], pool_size, n + 1);
        }
        if(n+2 < N){
            average_pool_kernel<<<gridDimPool, blockDimPool, 0, stream2>>>(deviceInputPool, deviceOutputPool, xdims[1], xdims[2], xdims[3], ydims[1], ydims[2], ydims[3], pool_size, n + 2);
        }
    }

    cudaDeviceSynchronize();

}

void setup_cuda_mem()
{
    // conv layer 1 vars
    const int adims[] = {xdims[0], (xdims[1] - conv1dims[0] + 1), (xdims[2] - conv1dims[1] + 1), conv1dims[3]};

    // avg pool 1 vars
    const int pool_size = 2;
    const int bdims[]   = {adims[0], adims[1] / pool_size, adims[2] / pool_size, adims[3]};

    // conv layer 2 vars
    const int cdims[] = {bdims[0], (bdims[1] - conv2dims[0] + 1), (bdims[2] - conv2dims[1] + 1), conv2dims[3]};

    // avg pool 2 vars
    const int ddims[] = {cdims[0], cdims[1] / pool_size, cdims[2] / pool_size, cdims[3]};

    // fully connected layer 1 vars
    const int ddims2[] = {ddims[0], ddims[1] * ddims[2] * ddims[3]};
    const int edims[] = {ddims[0], fc1dims[1]};

    // fully connected layer 2 vars
    const int fdims[] = {edims[0], fc2dims[1]};


    fullyForwardOut2 = zeros<float>(fdims);

    // forward_operation
    check_success(cudaMalloc((void**)&deviceOutputConv1, adims[0]*adims[1]*adims[2]*adims[3]*xdims[3]*sizeof(float)));
    check_success(cudaMalloc((void**)&deviceOutputPool1, bdims[0]*bdims[1]*bdims[2]*bdims[3]*xdims[3]*sizeof(float)));
    check_success(cudaMalloc((void**)&deviceOutputConv2, cdims[0]*cdims[1]*cdims[2]*cdims[3]*sizeof(float)));
    check_success(cudaMalloc((void**)&deviceOutputPool2, ddims[0]*ddims[1]*ddims[2]*ddims[3]*xdims[3]*sizeof(float)));
    check_success(cudaMalloc((void**)&deviceMaskFullyForward1, ddims2[1]*fc1dims[1]*sizeof(float)));
    check_success(cudaMalloc((void**)&deviceOutputFullyForward1, edims[0]*edims[1]*sizeof(float)));
    check_success(cudaMalloc((void**)&deviceMaskFullyForward2, edims[1]*fc2dims[1]*sizeof(float)));
    check_success(cudaMalloc((void**)&deviceOutputFullyForward2, fdims[0]*fdims[1]*sizeof(float)));

    // conv 1 layer
    int N  = xdims[0];
    int M = conv1dims[3];
    int C = conv1dims[2];
    int H = xdims[1];
    int W = xdims[2];
    int K = conv1dims[0];
    int W_out = W - K + 1;
    int H_out = H - K + 1;
    int H_unroll = C * K * K;
    int W_unroll = H_out * W_out;

    check_success(cudaMalloc((void**)&device_W_unrolled_L1, M*C*K*K * sizeof(float)));
    check_success(cudaMalloc((void**)&device_W_L1, M*C*K*K * sizeof(float)));
    check_success(cudaMalloc((void**)&device_X_unrolled0_L1, W_unroll * H_unroll * sizeof(float)));
    check_success(cudaMalloc((void**)&device_X_unrolled1_L1, W_unroll * H_unroll * sizeof(float)));
    check_success(cudaMalloc((void**)&device_X_unrolled2_L1, W_unroll * H_unroll * sizeof(float)));
    check_success(cudaMalloc((void**)&device_X0_L1, C * H * W * sizeof(float)));
    check_success(cudaMalloc((void**)&device_X1_L1, C * H * W * sizeof(float)));
    check_success(cudaMalloc((void**)&device_X2_L1, C * H * W * sizeof(float)));
    check_success(cudaMalloc((void**)&device_Y_unrolled_L1,  M*N*W_out*H_out* sizeof(float)));

    // conv 1 layer
    N  = bdims[0];
    M = conv2dims[3];
    C = conv2dims[2];
    H = bdims[1];
    W = bdims[2];
    K = conv2dims[0];
    W_out = W - K + 1;
    H_out = H - K + 1;
    H_unroll = C * K * K;
    W_unroll = H_out * W_out;

    check_success(cudaMalloc((void**)&device_W_unrolled_L2, M*C*K*K * sizeof(float)));
    check_success(cudaMalloc((void**)&device_W_L2, M*C*K*K * sizeof(float)));
    check_success(cudaMalloc((void**)&device_X_unrolled0_L2, W_unroll * H_unroll * sizeof(float)));
    check_success(cudaMalloc((void**)&device_X_unrolled1_L2, W_unroll * H_unroll * sizeof(float)));
    check_success(cudaMalloc((void**)&device_X_unrolled2_L2, W_unroll * H_unroll * sizeof(float)));
    check_success(cudaMalloc((void**)&device_Y_unrolled_L2,  M*N*W_out*H_out* sizeof(float)));
}

// Forward operation for the CNN, a combination of conv layer + average pooling
// + relu
void forward_operation(float *x, float *conv1, float *conv2, float *fc1,
                       float *fc2, int *out) {

    // conv layer 1 vars
    const int adims[] = {xdims[0], (xdims[1] - conv1dims[0] + 1),
                         (xdims[2] - conv1dims[1] + 1), conv1dims[3]};

    // avg pool 1 vars
    const int pool_size = 2;
    const int bdims[]   = {adims[0], adims[1] / pool_size, adims[2] / pool_size,
                           adims[3]};

    // conv layer 2 vars
    const int cdims[] = {bdims[0], (bdims[1] - conv2dims[0] + 1),
                         (bdims[2] - conv2dims[1] + 1), conv2dims[3]};

    // avg pool 2 vars
    const int ddims[] = {cdims[0], cdims[1] / pool_size, cdims[2] / pool_size,
                         cdims[3]};

    // fully connected layer 1 vars
    const int ddims2[] = {ddims[0], ddims[1] * ddims[2] * ddims[3]};
    const int edims[] = {ddims[0], fc1dims[1]};

    // fully connected layer 2 vars
    const int fdims[] = {edims[0], fc2dims[1]};

    /*********************************************** CONV 1 Layer ************************************************/
    // Done using unrolling and matrix-matrix multiplication
    convLayer_forward_streamed(xdims[0], conv1dims[3], conv1dims[2], xdims[1], xdims[2], conv1dims[0], x, conv1, deviceOutputConv1, adims, device_X0_L1, device_X1_L1, device_X2_L1, device_X_unrolled0_L1, device_X_unrolled1_L1, device_X_unrolled2_L1, device_W_L1, device_W_unrolled_L1, device_Y_unrolled_L1);

    // relu
    dim3 blockDimRELU1(CUDA_MAX_NUM_THREADS, 1, 1);
    dim3 gridDimRELU1((adims[0]*adims[1]*adims[2]*adims[3] - 1)/CUDA_MAX_NUM_THREADS + 1 , 1, 1);
    relu_gpu<<<gridDimRELU1, blockDimRELU1>>>(deviceOutputConv1, adims[0]*adims[1]*adims[2]*adims[3]);
    cudaDeviceSynchronize();

    deviceInputPool1 = deviceOutputConv1;

    /*********************************************** AVG POOL 1 Layer ************************************************/
    // kernel dims
    int N = adims[0];
    int M = conv1dims[3];
    int Z = ((bdims[2]-1)/TILE_WIDTH+1)*((bdims[1]-1)/TILE_WIDTH+1);

    // avg pool 1 launch
    average_pool_streamed(N, M, Z, pool_size, deviceInputPool1, deviceOutputPool1, adims, bdims);

    // avg pool memory freed
    cudaFree(deviceInputPool1);

    device_X_L2 = deviceOutputPool1;

    /*********************************************** CONV 2 Layer ************************************************/
    // Done using unrolling and matrix-matrix multiplication
    convLayer_forward_reg(xdims[0], conv2dims[3], conv2dims[2], bdims[1], bdims[2], conv2dims[0], conv2, deviceOutputConv2, cdims, device_X_L2, device_X_unrolled0_L2, device_X_unrolled1_L2, device_X_unrolled2_L2, device_W_L2,  device_W_unrolled_L2, device_Y_unrolled_L2);

    // relu
    dim3 blockDimRELU2(CUDA_MAX_NUM_THREADS, 1, 1);
    dim3 gridDimRELU2((cdims[0]*cdims[1]*cdims[2]*cdims[3] - 1)/CUDA_MAX_NUM_THREADS + 1 , 1, 1);
    relu_gpu<<<gridDimRELU2, blockDimRELU2>>>(deviceOutputConv2, cdims[0]*cdims[1]*cdims[2]*cdims[3]);
    cudaDeviceSynchronize();

    deviceInputPool2 = deviceOutputConv2;

    /*********************************************** AVG POOL 2 Layer ************************************************/
    // kernel dims
    N = cdims[0];
    M = conv2dims[3];
    Z = ((ddims[2]-1)/TILE_WIDTH+1)*((ddims[1]-1)/TILE_WIDTH+1);//adims[2]*adims[1];

    // Second average pool kernel launch
    average_pool_streamed(N, M, Z, pool_size, deviceInputPool2, deviceOutputPool2, cdims, ddims);

    // avg pool memory freed
    cudaFree(deviceInputPool2);

    /*********************************************** FULLY CONNECTED 1 Layer ************************************************/
    // allocate memory for device fully connected layer 1
    deviceInputFullyForward1 = deviceOutputPool2;

    // copy data to device
    check_success(cudaMemcpy(deviceMaskFullyForward1, fc1, ddims2[1]*fc1dims[1]*sizeof(float),cudaMemcpyHostToDevice));

    // Initialize the grid and block dimensions
    dim3 blockDimensionFF1(MULT_TW, MULT_TW, 1);
    dim3 gridDimensionFF1(ceil((1.0*edims[1])/MULT_TW), ceil((1.0*edims[0])/MULT_TW), 1);

    // Use tiled matrix multiplication for fc1 layer
    matrixMultiplyShared<<<gridDimensionFF1, blockDimensionFF1>>>(deviceInputFullyForward1, deviceMaskFullyForward1, deviceOutputFullyForward1, ddims2[0], ddims2[1],
                                                                  fc1dims[0], fc1dims[1], edims[0], edims[1]);
    cudaDeviceSynchronize();

    // freeing device memory for fc1 layer
    cudaFree(deviceInputFullyForward1);
    cudaFree(deviceMaskFullyForward1);

    // relu
    dim3 blockDimRELU3(CUDA_MAX_NUM_THREADS, 1, 1);
    dim3 gridDimRELU3((edims[0]*edims[1] - 1)/CUDA_MAX_NUM_THREADS + 1 , 1, 1);
    relu_gpu<<<gridDimRELU3, blockDimRELU3>>>(deviceOutputFullyForward1,  edims[0]*edims[1]);
    cudaDeviceSynchronize();

    /*********************************************** FULLY CONNECTED 2 Layer ************************************************/
    // allocate memory for device fully connected layer 1
    deviceInputFullyForward2 = deviceOutputFullyForward1;

    // copy data to device
    check_success(cudaMemcpy(deviceMaskFullyForward2, fc2, edims[1]*fc2dims[1]*sizeof(float),cudaMemcpyHostToDevice));

    // Initialize the grid and block dimensions
    dim3 blockDimensionFF2(MULT_TW, MULT_TW, 1);
    dim3 gridDimensionFF2(ceil((1.0*fdims[1])/MULT_TW), ceil((1.0*fdims[0])/MULT_TW), 1);

    // Use tiled matrix multiplication to implement fc2 layer
    matrixMultiplyShared<<<gridDimensionFF2, blockDimensionFF2>>>(deviceInputFullyForward2, deviceMaskFullyForward2, deviceOutputFullyForward2, edims[0], edims[1],
                                                                  fc2dims[0], fc2dims[1], fdims[0], fdims[1]);
    cudaDeviceSynchronize();

    // copy output data back from device
    check_success(cudaMemcpy(fullyForwardOut2, deviceOutputFullyForward2, fdims[0]*fdims[1]*sizeof(float), cudaMemcpyDeviceToHost));

    // freeing device memory for fc2 layer
    cudaFree(deviceInputFullyForward2);
    cudaFree(deviceMaskFullyForward2);
    cudaFree(deviceOutputFullyForward2);

    /*********************************************** GAUSSIAN Layer ************************************************/
    argmax(fullyForwardOut2, fdims, out);

    // freeing host buffers
    delete[] fullyForwardOut2;
}

int main(int argc, char **argv) {

    if (argc != 3 && argc != 4) {
        std::cerr << "\n"
                  << "This program performs the forward opertion step for "
            "Convolutional Neural Network(CNN).  "
            "Sample usage: \n"
                  << argv[0]
                  << " [../data/test10.hdf5] [../data/model.hdf5] [10]\n";
        return -1;
    }
    FLAGS_testdata = std::string(argv[1]);
    FLAGS_model    = std::string(argv[2]);
    if (argc == 3) {
        const std::map<std::string, int> default_batch_sizes{
            {"../data/test2.hdf5", 2},
            {"../data/test10.hdf5", 10},
            {"../data/test100.hdf5", 100},
            {"../data/testfull.hdf5", 10000}};
        const auto batch_size_in_map = default_batch_sizes.find(FLAGS_testdata);
        if (batch_size_in_map == default_batch_sizes.end()) {
            std::cerr << "\nERROR:: Unrecognized file " << FLAGS_testdata << " batch_size must be specified.\n";
            return -1;
        }
        FLAGS_batch_size = batch_size_in_map->second;
    } else if (argc == 4) {
        FLAGS_batch_size = atoi(argv[3]);
    }
    xdims[0] = FLAGS_batch_size;
    rdims[0] = FLAGS_batch_size;

    // Load data into x and y
    float *x = allocate<float>(xdims);
    float *y = allocate<float>(rdims);
    loadData(x, y);

    // Load model
    float *conv1 = allocate<float>(conv1dims);
    float *conv2 = allocate<float>(conv2dims);
    float *fc1   = allocate<float>(fc1dims);
    float *fc2   = allocate<float>(fc2dims);
    loadModel(conv1, conv2, fc1, fc2);


    // Perform foward opertion
    int *out = zeros<int>(FLAGS_batch_size);

    // abdul said we were allowed to move the mallocs outside of the time stamp
    setup_cuda_mem();

    // get start time
    const auto start = now();
    forward_operation(x, conv1, conv2, fc1, fc2, out);
    // get end time
    const auto end = now();

    // get elapsed time in milliseconds
    const auto elapsed =
        std::chrono::duration<double, std::milli>(end - start).count();

    // Get reference
    int *ref = zeros<int>(FLAGS_batch_size);
    argmax(y, rdims, ref);

    // Calculate correctness
    int num_correct = 0;
    for (const auto i : range(0, FLAGS_batch_size)) {
        if (out[i] == ref[i]) {
            num_correct++;
        }
    }
    std::cout << "Done with " << FLAGS_batch_size << " queries in "
              << "elapsed = " << elapsed << " milliseconds. Correctness: "
              << static_cast<float>(num_correct) / FLAGS_batch_size << "\n";

    delete[] x;
    delete[] y;
    delete[] conv1;
    delete[] conv2;
    delete[] fc1;
    delete[] fc2;
    delete[] out;
    delete[] ref;

    return 0;
}
