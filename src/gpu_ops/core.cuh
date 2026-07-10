#pragma once
#include <string_view>
#include <stdexcept>

#include <cufft.h>
#include <cufftXt.h>


namespace chirp {


void Log(std::string_view str);


enum class Status {
    Success = 0,
    Error = 1
};


Status handleCudaError(cudaError_t status);
Status handleCufftError(cufftResult status);


class CudaDeviceMemory {
public:
    CudaDeviceMemory(size_t size) : m_size(size) {
        auto status = handleCudaError(cudaMalloc(&m_buf, m_size));
        if (status != Status::Success) {
            throw std::runtime_error("Unable to allocate CUDA memory");
        }
    }
    ~CudaDeviceMemory() {
        handleCudaError(cudaFree(m_buf));
    }

private:
    void* m_buf;
    size_t m_size;
};



class ChirpSpectrumProcessor {
public:
    ChirpSpectrumProcessor(size_t n_series, size_t n_fft) : md_chirp(n_series), md_series(n_series), md_fft(n_fft) {
        handleCufftError(cufftCreate(&m_plan));
    };
    ~ChirpSpectrumProcessor() {
    }

private:
    cufftHandle m_plan;
    CudaDeviceMemory md_chirp;
    CudaDeviceMemory md_series;
    CudaDeviceMemory md_fft;
};


};  // namespace chirp

