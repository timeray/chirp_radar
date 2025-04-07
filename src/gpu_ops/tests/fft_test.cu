#include <complex>
#include <vector>
#include <cufft.h>
#include <gtest/gtest.h>

#include "gpu_ops/utils.cuh"


void test_chirp_filtering() {
    // Data
    size_t n_samples = 2048;
    float pulse_length_us = 900;
    float sampling_frequency_mhz = 1;

    size_t pulse_length_idx = static_cast<size_t>(pulse_length_us * sampling_frequency_mhz);    
    size_t n_delays = n_samples - pulse_length_idx;

    using data_type = std::complex<float>;

    std::vector<data_type> data(n_samples, 0);
    for (size_t i = 0; i < n_samples; ++i) {
        data[i] = data_type(i, -i);
    }

    size_t data_bytesize = sizeof(data_type) * data.size();

    // CUDA routines
    cufftHandle plan;
    size_t fft_size = 1024;

    check_cufft_error(cufftCreate(&plan));

    cufftComplex *d_data = nullptr;
    check_cuda_error(cudaMalloc(&d_data, data_bytesize));
    check_cuda_error(cudaMemcpy(d_data, data.data(), data_bytesize, cudaMemcpyHostToDevice));


    check_cufft_error(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));

    std::vector<data_type> data_fft(n_samples, 0);
    check_cuda_error(cudaMemcpy(data_fft.data(), d_data, data_bytesize, cudaMemcpyDeviceToHost));

    for (auto &val : data_fft) {
        printf("%f + %f j\n", val.real(), val.imag());
    }

    check_cuda_error(cudaFree(d_data));
    check_cufft_error(cufftDestroy(plan));
}


TEST(MainTestSuite, TestFFT) {
    test_chirp_filtering();
    ASSERT_TRUE(true);
}
