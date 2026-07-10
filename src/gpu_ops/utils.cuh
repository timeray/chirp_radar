#pragma once

#include <iostream>
#include <cufft.h>


// Returns the precision of the FFT transform for the given type
template <typename T>
constexpr cufftType fft_precision_v = CUFFT_C2C;

template <>
constexpr cufftType fft_precision_v<double> = CUFFT_Z2Z;


// CUDA API error checking
#define checkCudaError( call )                                                                                           \
    {                                                                                                                  \
        auto status = static_cast<cudaError_t>( call );                                                                \
        if ( status != cudaSuccess )                                                                                   \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "                                        \
                     "with "                                                                                           \
                     "%s (%d).\n",                                                                                     \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     cudaGetErrorString( status ),                                                                     \
                     status );                                                                                         \
    }

// cufft API error chekcing
#define checkCufftError( call )                                                                                             \
    {                                                                                                                  \
        auto status = static_cast<cufftResult>( call );                                                                \
        if ( status != CUFFT_SUCCESS )                                                                                 \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUFFT call \"%s\" in line %d of file %s failed "                                          \
                     "with "                                                                                           \
                     "code (%d).\n",                                                                                   \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     status );                                                                                         \
    }
/*
void checkCudaError(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cout << "CUDA ERROR: " << cudaGetErrorString(err) << std::endl;
    }
}


void checkCufftError(cufftResult_t res) {
    if (res != CUFFT_SUCCESS) {
        std::cout << "CUFFT ERROR: error code " << res << std::endl;
    }
}*/
