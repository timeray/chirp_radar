#pragma once
//#include <format>
#include <sstream>
#include <string_view>
#include <iostream>
#include <stdexcept>


namespace chirp {

void Log(std::string_view str) {
    std::cout << str << '\n';
}


enum class Status {
    Success = 0,
    Error = 1
};


Status handleCudaError(cudaError_t status) {
    if (status != cudaSuccess) {
        // auto str = std::format("CUDA RT call failed with {} (code {})", cudaGetErrorString(status), status);
        std::ostringstream os;
        os << "CUDA RT call failed with " << cudaGetErrorString(status);
        auto str = os.str();
        Log(str);
        return Status::Error;
    }
    return Status::Success;
}


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

};  // namespace chirp
