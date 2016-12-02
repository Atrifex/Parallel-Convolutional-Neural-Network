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

#define TILE_WIDTH 16
#define IO_LOGISTICS_SIZE 10

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

static int loadData(float *x, float *y) {
  // Open the data file
  const auto file_id = H5Fopen(FLAGS_testdata.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);

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
  check_success(H5Dread(x_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, x));
  check_success(H5Dread(y_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, y));

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
  check_success(H5Dread(conv1_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, conv1));
  check_success(H5Dread(conv2_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, conv2));
  check_success(H5Dread(fc1_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, fc1));
  check_success(H5Dread(fc2_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, fc2));

  // Close the dataset x and y
  check_success(H5Dclose(conv1_id));
  check_success(H5Dclose(conv2_id));
  check_success(H5Dclose(fc1_id));
  check_success(H5Dclose(fc2_id));

  // Close the file
  check_success(H5Fclose(file_id));
}

// CUDA kernel for forward convolution path
__global__ void conv_forward_kernel(float *deviceInput, float *deviceMask, float *deviceOutput, int IOLogistics[IO_LOGISTICS_SIZE])
{
    /*
     * This array stores:
     *  position 0: number of channels
     *  position 1: mask width
     *  position 2: mask height
     *  position 3: number of horizontal tiles per output map
     *  position 4: number of vertical tiles per output map
     *  position 5: Number of samples
     *  position 6: Input map height
     *  position 7: Input map width
     *  position 8: Number of output maps
     *  position 9: Number of input maps
     */

    int filter_h   = IOLogistics[1];
    int filter_w   = IOLogistics[2];
    int in_channel = IOLogistics[9];

    int n, m, h, w, c, p, q;
    n = blockIdx.x;
    m = blockIdx.y;
    h = blockIdx.z / (IOLogistics[4]) + threadIdx.y;
    w = blockIdx.z % (IOLogistics[3]) + threadIdx.x;
  
    float acc = 0.0f;
    for (int p : range(0, filter_h)) {
        for (int q : range(0, filter_w)) {
            for (int c : range(0, in_channel)) {
                int xoffset = n *IOLogistics[6]*IOLogistics[7]*IOLogistics[0] + (h + p)*IOLogistics[7]*IOLogistics[0] + (w + q)*IOLogistics[0] + c;
                int woffset = p * filter_w * in_channel * IOLogistics[8] + q*in_channel*IOLogistics[8] + c*IOLogistics[8] + m;
                acc += deviceInput[xoffset]*deviceMask[woffset];
            }
        }
    }
    int yoffset = ((n *(IOLogistics[6]-IOLogistics[2] + 1) + h) * (IOLogistics[7]-IOLogistics[1] + 1) + w)*IOLogistics[8] + m;
    deviceOutput[yoffset] = acc;
}

// From book chapter Figure 16.4
static void conv_forward_valid(const float *X, const int xdims[4],
                               const float *W, const int wdims[4], float *Y,
                               const int ydims[4]) {
  const auto filter_h   = wdims[0];
  const auto filter_w   = wdims[1];
  const auto in_channel = wdims[2];
  
  for (const auto i : range(0, ydims[0])) {
      for (const auto m : range(0, ydims[3])) {
          for (const auto w : range(0, ydims[2])) {
              for (const auto h : range(0, ydims[1])) {
                  for (const auto p : range(0, filter_h)) {
                      for (const auto q : range(0, filter_w)) {
                          for (const auto c : range(0, in_channel)) {
                              const auto yoffset = ((i * ydims[1] + h) * ydims[2] + w) * ydims[3] + m;
                              const auto xoffset = i * xdims[1] * xdims[2] * xdims[3] + (h + p) * xdims[2] * xdims[3] + (w + q) * xdims[3] + c;
                              const auto woffset = p * wdims[1] * wdims[2] * wdims[3] + q * wdims[2] * wdims[3] + c * wdims[3] + m;
                              Y[yoffset] += X[xoffset] * W[woffset];
                          }
                      }
                  }
              }
          }
      }
  }
}

// Recified linear unit 4d
static void relu4(float *X, const int xdims[4]) {
  for (const auto i : range(0, xdims[0] * xdims[1] * xdims[2] * xdims[3])) {
    X[i] = (X[i] < 0) ? 0 : X[i];
  }
}

// Recified linear unit 2d
static void relu2(float *X, const int xdims[2]) {
  for (const auto i : range(0, xdims[0] * xdims[1])) {
    X[i] = (X[i] < 0) ? 0 : X[i];
  }
}

// From book chapter Figure 16.5
static void average_pool(const float *X, const int xdims[4], const int pool_size, float *Y, const int ydims[4]) {
    
    for (const auto i : range(0, ydims[0])) {
        for (const auto m : range(0, ydims[3])) {
            for (const auto w : range(0, ydims[2])) {
                for (const auto h : range(0, ydims[1])) {
                    for (const auto p : range(0, pool_size)) {
                        for (const auto q : range(0, pool_size)) {
                            const auto yoffset = ((i * ydims[1] + h) * ydims[2] + w) * ydims[3] + m;
                            const auto xoffset = i * xdims[1] * xdims[2] * xdims[3] + (pool_size * h + p) * xdims[2] * xdims[3] + (pool_size * w + q) * xdims[3] + m;
                            Y[yoffset] += X[xoffset] / (1.0f * pool_size * pool_size);
                        }
                    }
                }
            }
        }
    }
    
}

static void fully_forward(const float *X, const int xdims[2], float *W, const int wdims[2], float *Y, const int ydims[2]) {

    for (const auto i : range(0, xdims[0])) {
        for (const auto j : range(0, wdims[1])) {
            float sum = 0;
            for (const auto k : range(0, xdims[1])) {
                sum += X[i * xdims[1] + k] * W[k * wdims[1] + j];
            }
            Y[i * wdims[1] + j] = sum;
        }
    }
}

// Choose the guess with largest score
static void argmax(const float *X, const int xdims[2], int *Y) {

    for (const auto i : range(0, xdims[0])) {
        auto max_idx = 0;
        auto max = X[i * xdims[1]];
        for (const auto j : range(0, xdims[1])) {
            const auto elem = X[(i * xdims[1]) + j];
            if (elem > max) {
                max_idx = j;
                max = elem;
            }
        }
        Y[i] = max_idx;
    }
}

// Forward operation for the CNN, a combination of conv layer + average pooling
// + relu
void forward_operation(float *x, float *conv1, float *conv2, float *fc1, float *fc2, int *out) {

    // conv layer
    const int adims[] = {xdims[0], (xdims[1] - conv1dims[0] + 1), (xdims[2] - conv1dims[1] + 1), conv1dims[3]};
    auto a = zeros<float>(adims);
    
    // average pooling
    const int pool_size = 2;
    const int bdims[]   = {adims[0], adims[1] / pool_size, adims[2] / pool_size, adims[3]};
    auto b = zeros<float>(bdims);

    // conv layer
    const int cdims[] = {bdims[0], (bdims[1] - conv2dims[0] + 1), (bdims[2] - conv2dims[1] + 1), conv2dims[3]};
    auto c = zeros<float>(cdims);

    // average pooling
    const int ddims[] = {cdims[0], cdims[1] / pool_size, cdims[2] / pool_size, cdims[3]};
    auto d = zeros<float>(ddims);
    
    // reshape
    const int ddims2[] = {ddims[0], ddims[1] * ddims[2] * ddims[3]};
    
    // matrix multiplication
    const int edims[] = {ddims[0], fc1dims[1]};
    auto e = zeros<float>(edims);
    
    // matrix multiplication
    const int fdims[] = {edims[0], fc2dims[1]};
    auto f = zeros<float>(fdims);

    // Memory instantiated on the GPU to hold input data, convolution mask, output data
    float * deviceForwardConvInput1, * deviceMask1, *deviceForwardConvOutput1;
    float * deviceForwardConvInput2, * deviceMask2, *deviceForwardConvOutput2;
    int * deviceIOLogistics;

    /***** Device data for input, mask, data dimensions *****/
    int outputMapWidth = xdims[2] - conv1dims[1] + 1;
    int outputMapHeight = xdims[1] - conv1dims[0] + 1;
    
    /*
     * This array stores:
     *  position 0: number of channels
     *  position 1: mask width
     *  position 2: mask height
     *  position 3: number of horizontal tiles per output map
     *  position 4: number of vertical tiles per output map
     *  position 5: Number of samples
     *  position 6: Input map height
     *  position 7: Input map width
     *  position 8: Number of output maps
     *  position 9: Number of input maps
     */
    int IOLogistics[IO_LOGISTICS_SIZE] = {xdims[3], conv1dims[1], conv1dims[0], (outputMapWidth-1)/TILE_WIDTH + 1, (outputMapHeight-1)/TILE_WIDTH + 1, xdims[0], xdims[1], xdims[2], conv1dims[3], conv1dims[2]};

    // Allocate host memory for all of the output maps generated by convolution 1
   	float * conv1Output = new float[IOLogistics[5]*outputMapWidth * outputMapHeight * conv1dims[3]*xdims[3]];
   	
   	// Allocate device memory
   	check_success(cudaMalloc((void**)&deviceMask1, xdims[3]*IOLogistics[1]*IOLogistics[2]*sizeof(float)));
   	check_success(cudaMalloc((void**)&deviceForwardConvInput1, IOLogistics[5]*xdims[3]*xdims[2]*xdims[1]*conv1dims[2]*sizeof(float)));
   	check_success(cudaMalloc((void**)&deviceForwardConvOutput1, IOLogistics[5]*xdims[3]*outputMapWidth*outputMapHeight*conv1dims[3]*sizeof(float)));
   	check_success(cudaMalloc((void**)&deviceIOLogistics, IO_LOGISTICS_SIZE*sizeof(int)));

    // perform memcopy for the first forward convolution
    check_success(cudaMemcpy(deviceMask1, conv1, xdims[3]*IOLogistics[1]*IOLogistics[2]*sizeof(float),cudaMemcpyHostToDevice));
    check_success(cudaMemcpy(deviceForwardConvInput1, x, IOLogistics[5]*xdims[3]*xdims[2]*xdims[1]*conv1dims[2]*sizeof(float),cudaMemcpyHostToDevice));
    check_success(cudaMemcpy(deviceIOLogistics, IOLogistics, IO_LOGISTICS_SIZE*sizeof(int),cudaMemcpyHostToDevice));
    
    /* 
     * Set grid and block dimensions for first kernel call. Each thread computes one element of one output feature map.
     * Each thread block computes output map elements for one tile.
     * Blocks are layered in a three-dimensional grid.
     * gridDim.x: Number of image samples
     * gridDim.y: Number of output feature maps
     * gridDim.z: Number of tiles per output map
     */
    dim3 blockDimConv1(TILE_WIDTH, TILE_WIDTH, 1);
  	dim3 gridDimConv1(xdims[0], conv1dims[3], IOLogistics[3]*IOLogistics[4]);

    // kernel launch for first convolution layer
    conv_forward_kernel<<<gridDimConv1, blockDimConv1>>>(deviceForwardConvInput1, deviceMask1, deviceForwardConvOutput1, deviceIOLogistics);
    cudaDeviceSynchronize();

    // copy output back
    cudaMemcpy(conv1Output, deviceForwardConvOutput1, IOLogistics[5]*xdims[3]*outputMapWidth*outputMapHeight*conv1dims[3]*sizeof(float), cudaMemcpyDeviceToHost);

    // conv 1
    conv_forward_valid(x, xdims, conv1, conv1dims, a, adims);

    for(int i = 0; i < IOLogistics[5]*xdims[3]*outputMapWidth*outputMapHeight*conv1dims[3]; i++){
        if(a[i] != conv1Output[i]){
            std::cout << "Failed on: " << i << std::endl;
        }
    }

    /// relu layer
    relu4(a, adims);

    // avg 1
    average_pool(a, adims, pool_size, b, bdims);
  
    // conv 2
    conv_forward_valid(b, bdims, conv2, conv2dims, c, cdims);

    /// relu layer
    relu4(c, cdims);

    // avg 2
    average_pool(c, cdims, pool_size, d, ddims);

    // full forward 1
    fully_forward(d, ddims2, fc1, fc1dims, e, edims);

    // relu
    relu2(e, edims);

    // full forward 2
    fully_forward(e, edims, fc2, fc2dims, f, fdims);
    
    // gaussian layer
    argmax(f, fdims, out);

    delete[] a;
    delete[] b;
    delete[] c;
    delete[] d;
    delete[] e;
    delete[] f;

    delete[] conv1Output;

    cudaFree(deviceMask1);
    cudaFree(deviceForwardConvInput1);
    cudaFree(deviceForwardConvOutput1);
    cudaFree(deviceIOLogistics);
}

int main(int argc, char **argv) {
    
    if (argc != 3 && argc != 4) {
        std::cerr << "\n"
                  << "This program performs the forward opertion step for "
                  << "Convolutional Neural Network(CNN).  "
                  << "Sample usage: \n"
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
    
    // get start time
    const auto start = now();
    
    forward_operation(x, conv1, conv2, fc1, fc2, out);
    
    // get end time
    const auto end = now();
    
    // get elapsed time in milliseconds
    const auto elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
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
