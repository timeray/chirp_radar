#pragma once

#include <vector>
#include <complex>
#include <type_traits>
#include <stdexcept>

#include <cufft.h>

#include <gpu_ops/utils.cuh>


template <typename T>
void simpleFFT1d(const std::vector<std::complex<T>>& series, std::vector<std::complex<T>>& out) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "T must be float or double");

    size_t data_size = sizeof(std::complex<T>) * series.size();

    // Precision of complex-to-complex transform
    cufftType fft_prec = CUFFT_C2C;
    if constexpr (std::is_same_v<T, double>) {
        fft_prec = CUFFT_Z2Z;
    }

    cufftHandle plan;
    checkCufftError(cufftCreate(&plan));
    checkCufftError(cufftPlan1d(&plan, series.size(), fft_prec, 1));

    using cufft_float_t = std::conditional_t<std::is_same_v<T, float>, cufftComplex, cufftDoubleComplex>;
    cufft_float_t* d_data = nullptr, *d_out = nullptr;
    checkCudaError(cudaMalloc(&d_data, data_size));
    checkCudaError(cudaMalloc(&d_out, data_size));

    checkCudaError(cudaMemcpy(d_data, series.data(), data_size, cudaMemcpyHostToDevice));

    if constexpr (std::is_same_v<T, float>) {
        checkCufftError(cufftExecC2C(plan, d_data, d_out, CUFFT_FORWARD));
    } else {
        checkCufftError(cufftExecZ2Z(plan, d_data, d_out, CUFFT_FORWARD));
    }

    checkCudaError(cudaMemcpy(out.data(), d_out, data_size, cudaMemcpyDeviceToHost));

    checkCudaError(cudaDeviceSynchronize());

    checkCudaError(cudaFree(d_data));
    checkCudaError(cudaFree(d_out));
    checkCufftError(cufftDestroy(plan));
}


template <typename T>
std::vector<std::complex<T>> movingFFT1d(const std::vector<std::complex<T>>& series, size_t n_fft) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "T must be float or double");

    if (n_fft > series.size()) {
        throw std::invalid_argument("FFT size should be less or equal to the size of input sequence");
    }
    size_t n_batches = series.size() - n_fft + 1;  // number of batched (transformation windows)
    size_t input_data_size = sizeof(std::complex<T>) * series.size();
    size_t output_data_size = sizeof(std::complex<T>) * n_fft * n_batches;

    // Precision of complex-to-complex transform
    cufftType fft_prec = CUFFT_C2C;
    if constexpr (std::is_same_v<T, double>) {
        fft_prec = CUFFT_Z2Z;
    }

    // Size in each dimension
    int n[] = {static_cast<int>(n_fft)};
    //int* inembed = NULL;
    //int* onembed = NULL;
    int inembed[] = {static_cast<int>(n_fft)};
    int onembed[] = {static_cast<int>(n_fft)};
    
    cufftHandle plan;
    checkCufftError(cufftCreate(&plan));
    checkCufftError(cufftPlanMany(
        &plan,
        1,          // 1D transform
        n,
        inembed,    // Input embedding (contiguous)
        1,          // Input stride
        1,          // Input distance
        onembed,    // Output embedding
        1,          // Output stride
        n_fft,      // Output distance
        fft_prec,
        n_batches
    ));

    // Output array (flattened matrix)
    std::vector<std::complex<T>> result(n_batches * n_fft);

    using cufft_float_t = std::conditional_t<std::is_same_v<T, float>, cufftComplex, cufftDoubleComplex>;
    cufft_float_t* d_data = nullptr, *d_out = nullptr;
    checkCudaError(cudaMalloc(&d_data, input_data_size));
    checkCudaError(cudaMalloc(&d_out, output_data_size));

    checkCudaError(cudaMemcpy(d_data, series.data(), input_data_size, cudaMemcpyHostToDevice));

    if constexpr (std::is_same_v<T, float>) {
        checkCufftError(cufftExecC2C(plan, d_data, d_out, CUFFT_FORWARD));
    } else {
        checkCufftError(cufftExecZ2Z(plan, d_data, d_out, CUFFT_FORWARD));
    }

    checkCudaError(cudaMemcpy(result.data(), d_out, output_data_size, cudaMemcpyDeviceToHost));

    checkCudaError(cudaDeviceSynchronize());

    checkCudaError(cudaFree(d_data));
    checkCudaError(cudaFree(d_out));
    checkCufftError(cufftDestroy(plan));

    return result;
}
