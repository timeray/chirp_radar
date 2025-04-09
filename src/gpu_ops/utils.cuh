#pragma once

#include <iostream>
#include <cufft.h>


void checkCudaError(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cout << "CUDA ERROR: " << cudaGetErrorString(err) << std::endl;
    }
}


void checkCufftError(cufftResult_t res) {
    if (res != CUFFT_SUCCESS) {
        std::cout << "CUFFT ERROR: error code " << res << std::endl;
    }
}
