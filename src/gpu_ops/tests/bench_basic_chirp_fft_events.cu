#include <complex>
#include <vector>
#include <cmath>
#include <type_traits>
#include <string>
#include <iostream>

#include <cufft.h>

#include "gpu_ops/tests/basic_chirp_fft.cuh"
#include "gpu_ops/utils.cuh"


template <typename T>
static void benchChirpFFT(size_t n_series, size_t n_fft, size_t n_iter = 1000) {
    // Setup
    using data_t = std::complex<T>;
    std::vector<data_t> series(n_series);
    std::vector<data_t> chirp(n_fft);

    for (size_t i = 0; i < n_series; ++i) {
        T v = static_cast<T>(i) / T(n_series);
        series[i] = data_t(v, -v);
        if (i < n_fft) {
            chirp[i] = data_t(v, -v);
        }
    }

    size_t n_batches = series.size() - n_fft + 1;  // number of batch (transformation) windows
    size_t input_data_size = sizeof(std::complex<T>) * series.size();
    size_t chirp_data_size = sizeof(std::complex<T>) * chirp.size();
    size_t output_data_size = sizeof(std::complex<T>) * n_fft * n_batches;

    constexpr cufftType fft_prec = fft_precision_v<T>;

    int rank = 1;
    int n[] = {static_cast<int>(n_fft)};
    int inembed[] = {static_cast<int>(n_fft)};
    int onembed[] = {static_cast<int>(n_fft)};

    gpu::CufftPlan plan_wrapper;
    CHECK_CUFFT(cufftPlanMany(&plan_wrapper.get(), rank, n, inembed, 1, n_fft, onembed, 1, n_fft, fft_prec, n_batches));

    // Output array (flattened matrix)
    std::vector<std::complex<T>> result(n_batches * n_fft);

    using cufft_float_t = std::conditional_t<std::is_same_v<T, float>, cufftComplex, cufftDoubleComplex>;
    gpu::CudaDeviceMemory d_data_mem(input_data_size);
    gpu::CudaDeviceMemory d_chirp_mem(chirp_data_size);
    gpu::CudaDeviceMemory d_out_mem(output_data_size);
    auto* d_data  = static_cast<cufft_float_t*>(d_data_mem.get());
    auto* d_chirp = static_cast<cufft_float_t*>(d_chirp_mem.get());
    auto* d_out   = static_cast<cufft_float_t*>(d_out_mem.get());

    // Setup callback
    using pars_t = ChirpCallbackParams<cufft_float_t>;
    pars_t host_params{d_chirp, n_fft};

    gpu::CudaDeviceMemory d_params_mem(sizeof(pars_t));
    auto* device_params = static_cast<pars_t*>(d_params_mem.get());
    CHECK_CUDA(cudaMemcpy(device_params, &host_params, sizeof(pars_t), cudaMemcpyHostToDevice));

    using cufft_cb_t = std::conditional_t<std::is_same_v<T, float>, cufftCallbackLoadC, cufftCallbackLoadZ>;
    cufft_cb_t load_callback_ptr;

    if constexpr (std::is_same_v<T, float>) {
        cudaMemcpyFromSymbol(&load_callback_ptr, chirpMultiplyCallbackCPtr, sizeof(load_callback_ptr));
        CHECK_CUFFT(cufftXtSetCallback(
            plan_wrapper.get(), (void**)&load_callback_ptr, CUFFT_CB_LD_COMPLEX, (void**)&device_params
        ));
    } else {
        cudaMemcpyFromSymbol(&load_callback_ptr, chirpMultiplyCallbackZPtr, sizeof(load_callback_ptr));
        CHECK_CUFFT(cufftXtSetCallback(
            plan_wrapper.get(), (void**)&load_callback_ptr, CUFFT_CB_LD_COMPLEX_DOUBLE, (void**)&device_params
        ));
    }

    // Measuring loop
    cudaEvent_t start, end;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&end));
    CHECK_CUDA(cudaEventRecord(start, 0));
    float elapsed_time = 0.0;

    for (size_t iter_num = 0; iter_num < n_iter; ++iter_num) {
        // This code get timed
        CHECK_CUDA(cudaMemcpy(d_data, series.data(), input_data_size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_chirp, chirp.data(), chirp_data_size, cudaMemcpyHostToDevice));

        // Execute FFT
        if constexpr (std::is_same_v<T, float>) {
            CHECK_CUFFT(cufftExecC2C(plan_wrapper.get(), d_data, d_out, CUFFT_FORWARD));
        } else {
            CHECK_CUFFT(cufftExecZ2Z(plan_wrapper.get(), d_data, d_out, CUFFT_FORWARD));
        }

        // No memcpy (assuming further processing is performed with GPU)
        // CHECK_CUDA(cudaMemcpy(result.data(), d_out, output_data_size, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    CHECK_CUDA(cudaEventRecord(end, 0));
    CHECK_CUDA(cudaEventSynchronize(end));
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, end));
    elapsed_time /= static_cast<float>(n_iter);

    std::string type_str;
    if constexpr (std::is_same_v<T, float>) {
        type_str = "float";
    } else {
        type_str = "double";
    }
    std::cout << "Benchmark for type = " << type_str << ", n_series = " << n_series << ", n_fft = " << n_fft << ". ";
    std::cout << "Elapsed time: " << elapsed_time << " ms\n";

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(end));

    // RAII destructors handle cleanup: cufftDestroy, cudaFree for all allocations
}


int main() {
    std::cout << "Dummy run\n";
    benchChirpFFT<float>(4096, 1024);
    std::cout << "Measurements\n";
    benchChirpFFT<float>(4096, 1024);
    benchChirpFFT<float>(16384, 1024);
    benchChirpFFT<double>(4096, 1024);
}
