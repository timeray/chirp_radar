#include <complex>
#include <vector>
#include <cmath>
#include <type_traits>
#include <string>
#include <iostream>

#include <fftw3.h>
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
        float_t v = static_cast<T>(i) / n_series;
        series[i] = data_t(v, -v);
        if (i < n_fft) {
            chirp[i] = data_t(v, -v);
        }
    }    

    size_t n_batches = series.size() - n_fft + 1;  // number of batch (transformation) windows
    size_t input_data_size = sizeof(std::complex<T>) * series.size();
    size_t chirp_data_size = sizeof(std::complex<T>) * chirp.size();
    size_t output_data_size = sizeof(std::complex<T>) * n_fft * n_batches;
    
    // Precision of complex-to-complex transform
    cufftType fft_prec = CUFFT_C2C;
    if constexpr (std::is_same_v<T, double>) {
        fft_prec = CUFFT_Z2Z;
    }

    int rank = 1;
    int n[] = {static_cast<int>(n_fft)};
    int inembed[] = {static_cast<int>(n_fft)};
    int onembed[] = {static_cast<int>(n_fft)};

    cufftHandle plan;
    checkCufftError(cufftCreate(&plan));
    checkCufftError(cufftPlanMany(&plan, rank, n, inembed, 1, n_fft, onembed, 1, n_fft, fft_prec, n_batches));

    // Output array (flattened matrix)
    std::vector<std::complex<T>> result(n_batches * n_fft);
    
    using cufft_float_t = std::conditional_t<std::is_same_v<T, float>, cufftComplex, cufftDoubleComplex>;
    cufft_float_t* d_data = nullptr, *d_chirp = nullptr, *d_out = nullptr;
    checkCudaError(cudaMalloc(&d_data, input_data_size));
    checkCudaError(cudaMalloc(&d_chirp, chirp_data_size));
    checkCudaError(cudaMalloc(&d_out, output_data_size));

    // Setup callback
    using pars_t = ChirpCallbackParams<cufft_float_t>;
    pars_t host_params{d_chirp, n_fft};
    pars_t* device_params;
    checkCudaError(cudaMalloc((void **)&device_params, sizeof(pars_t)));
    checkCudaError(cudaMemcpy(device_params, &host_params, sizeof(pars_t), cudaMemcpyHostToDevice));
 
    using cufft_cb_t = std::conditional_t<std::is_same_v<T, float>, cufftCallbackLoadC, cufftCallbackLoadZ>;
    cufft_cb_t load_callback_ptr;

    if constexpr (std::is_same_v<T, float>) {
        cudaMemcpyFromSymbol(&load_callback_ptr, chirpMultiplyCallbackCPtr, sizeof(load_callback_ptr));
        checkCufftError(cufftXtSetCallback(
            plan, (void**)&load_callback_ptr, CUFFT_CB_LD_COMPLEX, (void**)&device_params
        ));
    } else {
        cudaMemcpyFromSymbol(&load_callback_ptr, chirpMultiplyCallbackZPtr, sizeof(load_callback_ptr));
        checkCufftError(cufftXtSetCallback(
            plan, (void**)&load_callback_ptr, CUFFT_CB_LD_COMPLEX_DOUBLE, (void**)&device_params
        ));
    }

    // Measuring loop
    cudaEvent_t start, end;
    checkCudaError(cudaEventCreate(&start));
    checkCudaError(cudaEventCreate(&end));
    checkCudaError(cudaEventRecord(start, 0));
    float elapsed_time = 0.0;

    for (size_t iter_num = 0; iter_num < n_iter; ++iter_num) {
        // This code get timed
        checkCudaError(cudaMemcpy(d_data, series.data(), input_data_size, cudaMemcpyHostToDevice));
        checkCudaError(cudaMemcpy(d_chirp, chirp.data(), chirp_data_size, cudaMemcpyHostToDevice));

        // Execute FFT
        if constexpr (std::is_same_v<T, float>) {
            checkCufftError(cufftExecC2C(plan, d_data, d_out, CUFFT_FORWARD));
        } else {
            checkCufftError(cufftExecZ2Z(plan, d_data, d_out, CUFFT_FORWARD));
        }

        // No memcpy (assuming further processing is performed with GPU)
        // checkCudaError(cudaMemcpy(result.data(), d_out, output_data_size, cudaMemcpyDeviceToHost));
        checkCudaError(cudaDeviceSynchronize());
    }
    checkCudaError(cudaEventRecord(end, 0));
    checkCudaError(cudaEventSynchronize(end));
    checkCudaError(cudaEventElapsedTime(&elapsed_time, start, end));
    elapsed_time /= static_cast<float>(n_iter);
    
    std::string type_str;
    if constexpr (std::is_same_v<T, float>) {
        type_str = "float";
    } else {
        type_str = "double";
    }
    std::cout << "Benchmark for type = " << type_str << ", n_series = " << n_series << ", n_fft = " << n_fft << ". ";
    std::cout << "Elapsed time: " << elapsed_time << " ms\n";

    checkCudaError(cudaEventDestroy(start));
    checkCudaError(cudaEventDestroy(end));
    checkCudaError(cudaFree(device_params));
    checkCudaError(cudaFree(d_data));
    checkCudaError(cudaFree(d_out));
    checkCufftError(cufftDestroy(plan));
}


int main() {
    std::cout << "Dummy run\n";
    benchChirpFFT<float>(4096, 1024);
    std::cout << "Measurements\n";
    benchChirpFFT<float>(4096, 1024);
    benchChirpFFT<float>(16384, 1024);
    benchChirpFFT<double>(4096, 1024);
}

