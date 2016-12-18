#include <iostream>


using namespace std;


int main(int argc, char **argv) {
  int deviceCount;

  cudaGetDeviceCount(&deviceCount);

  for (int dev = 0; dev < deviceCount; dev++) {
    cudaDeviceProp deviceProp;

    cudaGetDeviceProperties(&deviceProp, dev);

    if (dev == 0) {
      if (deviceProp.major == 9999 && deviceProp.minor == 9999)
      {
        cout << "No CUDA GPU has been detected" << endl;
        return -1;
      }
      else if (deviceCount == 1)
      {
        //@@ WbLog is a provided logging API (similar to Log4J).
        //@@ The logging function wbLog takes a level which is either
        //@@ OFF, FATAL, ERROR, WARN, INFO, DEBUG, or TRACE and a
        //@@ message to be printed.
        cout << "There is 1 device supporting CUDA" << endl;
      }
      else
      {
        cout << "Device count: " << deviceCount << endl;
      }

    }

    cout << "Device " << dev << " name: "
         << deviceProp.name << endl;
    cout << " Computational Capabilities: "
         << deviceProp.major << "." << deviceProp.minor << endl;
    cout << " Maximum global memory size: "  << deviceProp.totalGlobalMem << endl;
    cout << " Maximum constant memory size: " << deviceProp.totalConstMem << endl;
    cout << " Maximum shared memory size per block: " << deviceProp.sharedMemPerBlock << endl;
    cout << " Maximum block dimensions: " << deviceProp.maxThreadsDim[0]
         << " x " << deviceProp.maxThreadsDim[1] << " x " << deviceProp.maxThreadsDim[2] << endl;
    cout << " Maximum grid dimensions: " << deviceProp.maxGridSize[0] << " x " << deviceProp.maxGridSize[1] << " x " <<
          deviceProp.maxGridSize[2] << endl;
    cout << " Warp size: " << deviceProp.warpSize << endl;
  }

  return 0;

}
