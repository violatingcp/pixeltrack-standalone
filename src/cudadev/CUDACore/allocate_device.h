#ifndef HeterogeneousCore_CUDAUtilities_allocate_device_h
#define HeterogeneousCore_CUDAUtilities_allocate_device_h

#include <cuda_runtime.h>

namespace cms {
  namespace cuda {
    // Allocate device memory on current device
    //void *allocate_device(size_t nbytes, cudaStream_t stream);
    void *allocate_device(int dev, size_t nbytes, cudaStream_t stream);

    // Free device memory (to be called from unique_ptr)
    //void free_device(void *ptr, cudaStream_t stream);
    void free_device(int device, void *ptr);
  }  // namespace cuda
}  // namespace cms

#endif
