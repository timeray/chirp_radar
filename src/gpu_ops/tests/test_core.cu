#include <gtest/gtest.h>

#include "gpu_ops/core.cuh"


TEST(MainTestSuite, TestCudaDeviceMemory) {
    gpu::CudaDeviceMemory(16);
}
