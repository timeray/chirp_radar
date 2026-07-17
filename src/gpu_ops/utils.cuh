#pragma once

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cufft.h>


// Returns the precision of the FFT transform for the given type
template <typename T>
constexpr cufftType fft_precision_v = CUFFT_C2C;

template <>
constexpr cufftType fft_precision_v<double> = CUFFT_Z2Z;


enum class ErrorPolicy {
    kLog,      // print and continue
    kThrow,    // throw std::runtime_error
    kAbort     // print and std::abort()
};

inline std::string errorString(cudaError_t status) {
    return cudaGetErrorString(status);
}

inline std::string errorString(cufftResult status) {
    return "code (" + std::to_string(static_cast<int>(status)) + ")";
}

template <typename ErrorT>
constexpr ErrorT successValue = static_cast<ErrorT>(0);

template <>
constexpr cudaError_t successValue<cudaError_t> = cudaSuccess;

template <>
constexpr cufftResult successValue<cufftResult> = CUFFT_SUCCESS;

template <typename ErrorT>
void check(ErrorT status, const char* call, const char* file, int line,
           ErrorPolicy policy = ErrorPolicy::kThrow) {
    if (status == successValue<ErrorT>)
        return;

    std::ostringstream os;
    os << "ERROR: " << call << " at " << file << ":" << line
       << " failed with " << errorString(status);

    switch (policy) {
        case ErrorPolicy::kLog:
            std::cerr << os.str() << '\n';
            break;
        case ErrorPolicy::kAbort:
            std::cerr << os.str() << '\n';
            std::abort();
        case ErrorPolicy::kThrow:
            throw std::runtime_error(os.str());
    }
}

#define CHECK_CUDA(call, ...)   check(static_cast<cudaError_t>(call),  #call, __FILE__, __LINE__, ##__VA_ARGS__)
#define CHECK_CUFFT(call, ...)  check(static_cast<cufftResult>(call), #call, __FILE__, __LINE__, ##__VA_ARGS__)
