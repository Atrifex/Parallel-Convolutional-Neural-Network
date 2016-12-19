#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <map>
#include <sys/time.h>
#include <valarray>

#include "range.hpp"
#include "utils.hpp"

#define NUM_ROWS 28
#define NUM_COLS 28
#define NUM_CHANNELS 1
#define NUM_DIGITS 10

using namespace std;

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
    const int adims[] = {xdims[0], (xdims[1] - conv1dims[0] + 1), (xdims[2] - conv1dims[1] + 1), conv1dims[3]};
    auto a = (float*)calloc(adims[0]*adims[1]*adims[2]*adims[3], sizeof(float));
    const int pool_size = 2;
    const int bdims[]   = {adims[0], adims[1] / pool_size, adims[2] / pool_size, adims[3]};
    auto b = (float*)calloc(bdims[0]*bdims[1]*bdims[2]*bdims[3], sizeof(float));
    const int cdims[] = {bdims[0], (bdims[1] - conv2dims[0] + 1), (bdims[2] - conv2dims[1] + 1), conv2dims[3]};
    auto c = (float*)calloc(cdims[0]*cdims[1]*cdims[2]*cdims[3], sizeof(float));
    const int ddims[] = {cdims[0], cdims[1] / pool_size, cdims[2] / pool_size, cdims[3]};
    auto d = (float*)calloc(ddims[0]*ddims[1]*ddims[2]*ddims[3], sizeof(float));
    // reshape
    const int ddims2[] = {ddims[0], ddims[1] * ddims[2] * ddims[3]};
    
    const int edims[] = {ddims[0], fc1dims[1]};
    auto e = (float*)calloc(edims[0]*edims[1], sizeof(float));
    const int fdims[] = {edims[0], fc2dims[1]};
    auto f = (float*)calloc(fdims[0]*fdims[1], sizeof(float));




    // conv layer 1
    conv_forward_valid(x, xdims, conv1, conv1dims, a, adims);

    /// relu layer
    relu4(a, adims);

    // average pooling 1
    average_pool(a, adims, pool_size, b, bdims);

    // conv layer 2
    conv_forward_valid(b, bdims, conv2, conv2dims, c, cdims);

    // relu
    relu4(c, cdims);

    // average pooling 2
    average_pool(c, cdims, pool_size, d, ddims);

    // fully forward layer 1
    fully_forward(d, ddims2, fc1, fc1dims, e, edims);

    // relu
    relu2(e, edims);

    // fully forward layer 2
    fully_forward(e, edims, fc2, fc2dims, f, fdims);

    argmax(f, fdims, out);



    delete[] a;
    delete[] b;
    delete[] c;
    delete[] d;
    delete[] e;
    delete[] f;
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

    // Load model
    float *conv1 = allocate<float>(conv1dims);
    float *conv2 = allocate<float>(conv2dims);
    float *fc1   = allocate<float>(fc1dims);
    float *fc2   = allocate<float>(fc2dims);

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
