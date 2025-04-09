#pragma once

#include <vector>
#include <complex>

#include <cufft.h>

#include <gpu_ops/utils.cuh>


template <typename T>
void chirpFFT1d(const std::vector<std::complex<T>>& series,
                const std::vector<std::complex<T>>& chirp,
                std::vector<std::complex<T>>& out) {
    size_t data_size = sizeof(std::complex<T>) * series.size();

    cufftHandle plan;

    checkCufftError(cufftCreate(&plan));
    checkCufftError(cufftPlan1d(&plan, series.size(), CUFFT_C2C, 1));

    cufftComplex* d_data = nullptr, *d_out = nullptr;
    checkCudaError(cudaMalloc(&d_data, data_size));
    checkCudaError(cudaMalloc(&d_out, data_size));

    checkCudaError(cudaMemcpy(d_data, series.data(), data_size, cudaMemcpyHostToDevice));

    checkCufftError(cufftExecC2C(plan, d_data, d_out, CUFFT_FORWARD));

    checkCudaError(cudaMemcpy(out.data(), d_out, data_size, cudaMemcpyDeviceToHost));

    checkCudaError(cudaDeviceSynchronize());

    checkCudaError(cudaFree(d_data));
    checkCudaError(cudaFree(d_out));
    checkCufftError(cufftDestroy(plan));
}
