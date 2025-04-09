#include <complex>
#include <vector>
#include <cufft.h>

#include <gtest/gtest.h>
#include <fftw3.h>

#include "gpu_ops/core.cuh"
#include "gpu_ops/utils.cuh"


TEST(MainTestSuite, TestFFT) {
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

    chirpFFT1d(series, chirp, out);

    // Check against FFTW3
    std::vector<data_t> out_fftw3(n_fft);
    fftwf_plan p;
    p = fftwf_plan_dft_1d(
        n_fft,
        reinterpret_cast<fftwf_complex*>(series.data()),
        reinterpret_cast<fftwf_complex*>(out_fftw3.data()),
        FFTW_FORWARD, FFTW_ESTIMATE
    );
    fftwf_execute(p);
    fftwf_destroy_plan(p);

    for (size_t i = 0; i < n_fft; ++i) {
        EXPECT_NEAR(out[i].real(), out_fftw3[i].real(), 1e-5f) 
            << "Vectors differ at [" << i << "] (CUDA FFT = " << out[i] << " vs FFTW3 = " << out_fftw3[i] << ")";
        EXPECT_NEAR(out[i].imag(), out_fftw3[i].imag(), 1e-5f) 
            << "Vectors differ at [" << i << "] (CUDA FFT = " << out[i] << " vs FFTW3 = " << out_fftw3[i] << ")";
    }
}
