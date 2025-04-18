#include <iostream>
//#include <format>
#include <sstream>

#include <cufft.h>

#include <gpu_ops/core.cuh>


namespace chirp {


void Log(std::string_view str) {
    std::cout << str << '\n';
}


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


};  // namespace chirp

