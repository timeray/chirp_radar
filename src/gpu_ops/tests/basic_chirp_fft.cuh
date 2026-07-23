#pragma once

#include <vector>
#include <complex>
#include <type_traits>

#include <cufft.h>
#include <cufftXt.h>

#include <gpu_ops/utils.cuh>
#include <gpu_ops/core.cuh>


// ---------------------------------------------------------------------------
// simpleFFT1d — single 1D FFT
// ---------------------------------------------------------------------------

// Device overload: output to pre-allocated device memory of size series.size()
template <typename T>
void simpleFFT1d(const std::vector<std::complex<T>>& series,
                 gpu::CudaDeviceMemory& d_out) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "T must be float or double");

    size_t data_size = sizeof(std::complex<T>) * series.size();
    using cufft_float_t = std::conditional_t<std::is_same_v<T, float>, cufftComplex, cufftDoubleComplex>;
    constexpr cufftType fft_prec = fft_precision_v<T>;

    gpu::CufftPlan plan_wrapper;
    CHECK_CUFFT(cufftPlan1d(&plan_wrapper.get(), series.size(), fft_prec, 1));

    gpu::CudaDeviceMemory d_data_mem(data_size);
    auto* data = static_cast<cufft_float_t*>(d_data_mem.get());
    auto* out  = static_cast<cufft_float_t*>(d_out.get());

    CHECK_CUDA(cudaMemcpy(data, series.data(), data_size, cudaMemcpyHostToDevice));

    if constexpr (std::is_same_v<T, float>) {
        CHECK_CUFFT(cufftExecC2C(plan_wrapper.get(), data, out, CUFFT_FORWARD));
    } else {
        CHECK_CUFFT(cufftExecZ2Z(plan_wrapper.get(), data, out, CUFFT_FORWARD));
    }

    CHECK_CUDA(cudaDeviceSynchronize());
}

// Host overload: copies result back to host vector
template <typename T>
std::vector<std::complex<T>> simpleFFT1d(const std::vector<std::complex<T>>& series) {
    size_t data_size = sizeof(std::complex<T>) * series.size();
    std::vector<std::complex<T>> out(series.size());
    gpu::CudaDeviceMemory d_out(data_size);
    simpleFFT1d(series, d_out);
    CHECK_CUDA(cudaMemcpy(out.data(), d_out.get(), data_size, cudaMemcpyDeviceToHost));
    return out;
}


// ---------------------------------------------------------------------------
// movingFFT1d — sliding-window batched FFT
// ---------------------------------------------------------------------------

// Returns the output size for the given series length and FFT window size
inline size_t movingFFT1dOutputSize(size_t n_series, size_t n_fft) {
    return (n_series - n_fft + 1) * n_fft;
}

// Device overload: output to pre-allocated device memory of size outputSize * sizeof(complex<T>)
template <typename T>
void movingFFT1d(const std::vector<std::complex<T>>& series, size_t n_fft,
                 gpu::CudaDeviceMemory& d_out) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "T must be float or double");

    if (n_fft > series.size()) {
        throw std::invalid_argument("FFT size should be less or equal to the size of input sequence");
    }
    size_t n_batches = series.size() - n_fft + 1;  // number of batch (transformation) windows
    size_t input_data_size = sizeof(std::complex<T>) * series.size();

    constexpr cufftType fft_prec = fft_precision_v<T>;

    // Size in each dimension
    int n[] = {static_cast<int>(n_fft)};
    int inembed[] = {static_cast<int>(n_fft)};
    int onembed[] = {static_cast<int>(n_fft)};

    gpu::CufftPlan plan_wrapper;
    CHECK_CUFFT(cufftPlanMany(
        &plan_wrapper.get(),
        1,          // 1D transform
        n,          // Shape of transform
        inembed,    // Shape of input data of fft
        1,          // Input stride
        1,          // Input distance
        onembed,    // Size of output data of fft
        1,          // Output stride
        n_fft,      // Output distance
        fft_prec,
        n_batches
    ));

    using cufft_float_t = std::conditional_t<std::is_same_v<T, float>, cufftComplex, cufftDoubleComplex>;
    gpu::CudaDeviceMemory d_data_mem(input_data_size);
    auto* data = static_cast<cufft_float_t*>(d_data_mem.get());
    auto* out  = static_cast<cufft_float_t*>(d_out.get());

    CHECK_CUDA(cudaMemcpy(data, series.data(), input_data_size, cudaMemcpyHostToDevice));

    if constexpr (std::is_same_v<T, float>) {
        CHECK_CUFFT(cufftExecC2C(plan_wrapper.get(), data, out, CUFFT_FORWARD));
    } else {
        CHECK_CUFFT(cufftExecZ2Z(plan_wrapper.get(), data, out, CUFFT_FORWARD));
    }

    CHECK_CUDA(cudaDeviceSynchronize());
}

// Host overload: allocates device memory, computes, copies result back
template <typename T>
std::vector<std::complex<T>> movingFFT1d(const std::vector<std::complex<T>>& series, size_t n_fft) {
    size_t n_batches = series.size() - n_fft + 1;
    size_t output_data_size = sizeof(std::complex<T>) * n_batches * n_fft;

    std::vector<std::complex<T>> result(n_batches * n_fft);
    gpu::CudaDeviceMemory d_out(output_data_size);
    movingFFT1d(series, n_fft, d_out);
    CHECK_CUDA(cudaMemcpy(result.data(), d_out.get(), output_data_size, cudaMemcpyDeviceToHost));
    return result;
}


// ---------------------------------------------------------------------------
// chirp FFT helpers (callback kernel + params)
// ---------------------------------------------------------------------------

template <typename cufftComplexT>
__device__ __host__ cufftComplexT complexConjMul(cufftComplexT a, cufftComplexT b) {
    return {a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y};
}


template <typename cufftComplexT>
struct ChirpCallbackParams {
    cufftComplexT* chirp;
    size_t n_fft;
};


template <typename cufftComplexT>
__device__ cufftComplexT kerChirpMultiplyLoadCallback(void* data_in, size_t offset,
                                                      void* caller_info, void* shared_ptr) {
    using pars_t = const ChirpCallbackParams<cufftComplexT>;
    pars_t* params = static_cast<pars_t*>(caller_info);
    cufftComplexT* series = static_cast<cufftComplexT*>(data_in);
    cufftComplexT* chirp = params->chirp;
    size_t n_fft = params->n_fft;

    size_t batch_index = static_cast<float>(offset) / static_cast<float>(n_fft);
    size_t batch_offset = offset % n_fft;
    return complexConjMul(series[batch_index + batch_offset], chirp[batch_offset]);
}


__device__ cufftCallbackLoadC chirpMultiplyCallbackCPtr = kerChirpMultiplyLoadCallback<cufftComplex>;
__device__ cufftCallbackLoadZ chirpMultiplyCallbackZPtr = kerChirpMultiplyLoadCallback<cufftDoubleComplex>;


// ---------------------------------------------------------------------------
// chirpFFT — sliding-window FFT with on-load chirp multiplication
// ---------------------------------------------------------------------------

// Returns the output size for the given series length and chirp (FFT window) size
inline size_t chirpFFTOutputSize(size_t n_series, size_t n_fft) {
    return (n_series - n_fft + 1) * n_fft;
}

// Device overload: output to pre-allocated device memory of size outputSize * sizeof(complex<T>)
template <typename T>
void chirpFFT(const std::vector<std::complex<T>>& series,
              const std::vector<std::complex<T>>& chirp,
              gpu::CudaDeviceMemory& d_out) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "T must be float or double");

    size_t n_fft = chirp.size();
    if (n_fft > series.size()) {
        throw std::invalid_argument("FFT size should be less or equal to the size of input sequence");
    }
    size_t n_batches = series.size() - n_fft + 1;  // number of batch (transformation) windows
    size_t input_data_size = sizeof(std::complex<T>) * series.size();
    size_t chirp_data_size = sizeof(std::complex<T>) * chirp.size();

    constexpr cufftType fft_prec = fft_precision_v<T>;

    int rank = 1;
    int n[] = {static_cast<int>(n_fft)};
    int inembed[] = {static_cast<int>(n_fft)};
    int onembed[] = {static_cast<int>(n_fft)};

    gpu::CufftPlan plan_wrapper;
    CHECK_CUFFT(cufftPlanMany(&plan_wrapper.get(), rank, n, inembed, 1, n_fft, onembed, 1, n_fft, fft_prec, n_batches));

    using cufft_float_t = std::conditional_t<std::is_same_v<T, float>, cufftComplex, cufftDoubleComplex>;
    gpu::CudaDeviceMemory d_data_mem(input_data_size);
    gpu::CudaDeviceMemory d_chirp_mem(chirp_data_size);
    auto* data  = static_cast<cufft_float_t*>(d_data_mem.get());
    auto* chirp_dev = static_cast<cufft_float_t*>(d_chirp_mem.get());
    auto* out   = static_cast<cufft_float_t*>(d_out.get());

    CHECK_CUDA(cudaMemcpy(data, series.data(), input_data_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(chirp_dev, chirp.data(), chirp_data_size, cudaMemcpyHostToDevice));

    // Setup callback parameters on device
    using pars_t = ChirpCallbackParams<cufft_float_t>;
    pars_t host_params{chirp_dev, n_fft};

    gpu::CudaDeviceMemory d_params_mem(sizeof(pars_t));
    auto* device_params = static_cast<pars_t*>(d_params_mem.get());
    CHECK_CUDA(cudaMemcpy(device_params, &host_params, sizeof(pars_t), cudaMemcpyHostToDevice));

    using cufft_cb_t = std::conditional_t<std::is_same_v<T, float>, cufftCallbackLoadC, cufftCallbackLoadZ>;
    cufft_cb_t load_callback_ptr;

    // Execute FFT
    if constexpr (std::is_same_v<T, float>) {
        cudaMemcpyFromSymbol(&load_callback_ptr, chirpMultiplyCallbackCPtr, sizeof(load_callback_ptr));
        CHECK_CUFFT(cufftXtSetCallback(
            plan_wrapper.get(), (void**)&load_callback_ptr, CUFFT_CB_LD_COMPLEX, (void**)&device_params
        ));
        CHECK_CUFFT(cufftExecC2C(plan_wrapper.get(), data, out, CUFFT_FORWARD));
    } else {
        cudaMemcpyFromSymbol(&load_callback_ptr, chirpMultiplyCallbackZPtr, sizeof(load_callback_ptr));
        CHECK_CUFFT(cufftXtSetCallback(
            plan_wrapper.get(), (void**)&load_callback_ptr, CUFFT_CB_LD_COMPLEX_DOUBLE, (void**)&device_params
        ));
        CHECK_CUFFT(cufftExecZ2Z(plan_wrapper.get(), data, out, CUFFT_FORWARD));
    }

    CHECK_CUDA(cudaDeviceSynchronize());
}

// Host overload: allocates device memory, computes, copies result back
template <typename T>
std::vector<std::complex<T>> chirpFFT(const std::vector<std::complex<T>>& series,
                                      const std::vector<std::complex<T>>& chirp) {
    size_t n_batches = series.size() - chirp.size() + 1;
    size_t n_fft = chirp.size();
    size_t output_data_size = sizeof(std::complex<T>) * n_batches * n_fft;

    std::vector<std::complex<T>> result(n_batches * n_fft);
    gpu::CudaDeviceMemory d_out(output_data_size);
    chirpFFT(series, chirp, d_out);
    CHECK_CUDA(cudaMemcpy(result.data(), d_out.get(), output_data_size, cudaMemcpyDeviceToHost));
    return result;
}
