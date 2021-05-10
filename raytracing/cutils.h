#ifndef CG_CUTILS_H_
#define CG_CUTILS_H_

#include <iostream>

namespace cg
{

constexpr int ERRNO_CUDA = 99;

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) checkCudaErrors_cuda( (val), #val, __FILE__, __LINE__ )

void checkCudaErrors_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "'" << std::endl;
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(ERRNO_CUDA);
    }
}

void initDevice(int const devNum)
{
    int dev = devNum;
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
    std::cout << "Using device " << dev << ": " << deviceProp.name << std::endl;
    checkCudaErrors(cudaSetDevice(dev));
}

} /* namespace cg */

#endif /* CG_CUTILS_H_ */
