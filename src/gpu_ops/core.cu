#include <stdio.h>
#include <gpu_ops/core.cuh>
#include <span>
#include <complex>
#include <vector>
#include <iostream>

#include <cufft.h>


__global__ void gpuPrint() {
    printf("Hello, GPU-accelarated world!\n");
}


void testGpuPrint() {
    gpuPrint<<<1, 1>>>();
    cudaDeviceSynchronize();
}


template <typename T>
/* CUDA 12.0 (C++20)
void chirpShiftedFFT(std::span<std::complex<T>> series, std::span<std::complex<T>> chirp, std::span<std::complex<T>> out) {
*/
void chirpShiftedFFT(const std::vector<std::complex<T>>& series,
                     const std::vector<std::complex<T>>& chirp,
                     std::vector<std::complex<T>> out) {
    size_t data_size = sizeof(std::complex<T>) * series.size();

    cufftHandle plan;

    cufftCreate(&plan);
    cufftPlan1d(&plan, series.size(), CUFFT_C2C, 1);

    cufftComplex* d_data = nullptr, *d_out = nullptr;
    cudaMalloc(&d_data, data_size);
    cudaMalloc(&d_out, data_size);
    cudaMemcpy(d_data, series.data(), data_size, cudaMemcpyHostToDevice);
    
    cufftResult res = cufftExecC2C(plan, d_data, d_out, CUFFT_FORWARD);
    if (res != CUFFT_SUCCESS) {
        std::cout << "Error: " << res << std::endl;
    }

    cudaMemcpy(out.data(), d_out, data_size, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    
    cudaFree(d_data);
    cudaFree(d_out);
    cufftDestroy(plan);
}


/* CUDA 12.0 (C++20)
template <typename T>
void chirpDemodulation(std::span<T> series, std::span<T> chirp) {
    // Get FFT of the product between series and conjugated chirp
    

    // Estimate delays of each object in spectrum
}
*/



void testChirpDemodulation() {
    size_t n_fft = 16;

    using data_t = std::complex<float>;
    std::vector<data_t> series(n_fft);
    std::vector<data_t> chirp(n_fft);
    std::vector<data_t> out(n_fft);

    for (size_t i = 0; i < n_fft; ++i) {
        float v = static_cast<float>(i);
        series[i] = data_t(v, -v);
        chirp[i] = data_t(v, -v);
    }
    

    chirpShiftedFFT(series, chirp, out);

    std::cout << "series:";
    for (size_t i = 0; i < n_fft; ++i) {
        std::cout << " " << series[i];
    }
    std::cout << "\n";
    std::cout << "chirp:";
    for (size_t i = 0; i < n_fft; ++i) {
        std::cout << " " << chirp[i];
    }
    std::cout << "\n";
    std::cout << "series:";
    for (size_t i = 0; i < n_fft; ++i) {
        std::cout << " " << out[i];
    }
    std::cout << "\n";
}
