#include <gtest/gtest.h>

#include "gpu_ops/core.cuh"


TEST(MainTestSuite, TestCudaDeviceMemory) {
    chirp::CudaDeviceMemory(16);
}

