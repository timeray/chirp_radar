#pragma once
#include <cstddef>
#include <cufft.h>
#include <cufftXt.h>

#include <gpu_ops/utils.cuh>


// GPU resource management
namespace gpu {


class CudaDeviceMemory {
public:
    CudaDeviceMemory() = delete;

    explicit CudaDeviceMemory(size_t size) : m_size(size) {
        CHECK_CUDA(cudaMalloc(&m_buf, m_size));
    }

    ~CudaDeviceMemory() {
        if (m_buf) {
            CHECK_CUDA(cudaFree(m_buf), ErrorPolicy::kLog);
        }
    }

    CudaDeviceMemory(CudaDeviceMemory&& other) noexcept
        : m_buf(other.m_buf), m_size(other.m_size) {
        other.m_buf = nullptr;
        other.m_size = 0;
    }

    CudaDeviceMemory& operator=(CudaDeviceMemory&& other) noexcept {
        if (this != &other) {
            if (m_buf) {
                CHECK_CUDA(cudaFree(m_buf), ErrorPolicy::kLog);
            }
            m_buf = other.m_buf;
            m_size = other.m_size;
            other.m_buf = nullptr;
            other.m_size = 0;
        }
        return *this;
    }

    CudaDeviceMemory(const CudaDeviceMemory&) = delete;
    CudaDeviceMemory& operator=(const CudaDeviceMemory&) = delete;

    void* get()             { return m_buf; }
    const void* get() const { return m_buf; }
    size_t size() const     { return m_size; }

private:
    void*  m_buf  = nullptr;
    size_t m_size = 0;
};


class CufftPlan {
public:
    CufftPlan() {
        CHECK_CUFFT(cufftCreate(&m_plan));
    }

    ~CufftPlan() {
        if (m_plan) {
            CHECK_CUFFT(cufftDestroy(m_plan), ErrorPolicy::kLog);
        }
    }

    CufftPlan(CufftPlan&& other) noexcept : m_plan(other.m_plan) {
        other.m_plan = 0;
    }

    CufftPlan& operator=(CufftPlan&& other) noexcept {
        if (this != &other) {
            if (m_plan) {
                CHECK_CUFFT(cufftDestroy(m_plan), ErrorPolicy::kLog);
            }
            m_plan = other.m_plan;
            other.m_plan = 0;
        }
        return *this;
    }

    CufftPlan(const CufftPlan&) = delete;
    CufftPlan& operator=(const CufftPlan&) = delete;

    cufftHandle&       get()       { return m_plan; }
    const cufftHandle& get() const { return m_plan; }

private:
    cufftHandle m_plan = 0;
};


}  // namespace gpu


// Radar processing

namespace chirp {


class ChirpSpectrumProcessor {
public:
    ChirpSpectrumProcessor(size_t n_series, size_t n_fft)
        : md_chirp(n_series), md_series(n_series), md_fft(n_fft) {}
    ~ChirpSpectrumProcessor() = default;

    ChirpSpectrumProcessor(const ChirpSpectrumProcessor&) = delete;
    ChirpSpectrumProcessor& operator=(const ChirpSpectrumProcessor&) = delete;

private:
    gpu::CufftPlan m_plan;
    gpu::CudaDeviceMemory md_chirp;
    gpu::CudaDeviceMemory md_series;
    gpu::CudaDeviceMemory md_fft;
};


};  // namespace chirp
