#include <complex>
#include <vector>
#include <cufft.h>
#include <cmath>

#include <gtest/gtest.h>
#include <fftw3.h>

#include "gpu_ops/core.cuh"
#include "gpu_ops/utils.cuh"



template <typename T>
::testing::AssertionResult isClose(T val1, T val2, T rel_err, T abs_err) {
    T diff = std::abs(val1 - val2);
    T lowest = val1 < val2 ? val1 : val2;
    T rel_diff = diff / lowest;
    if ((rel_diff < rel_err) || (diff < abs_err)) {
        return ::testing::AssertionSuccess();
    } else {
        return ::testing::AssertionFailure()
            << "\nThe relative difference is " << rel_diff << " vs " << rel_err
            << "\nThe absolute difference is " << diff <<  " vs " << abs_err
            << "\nval1 = " << val1 << "\n" << "val2 = " << val2 << "\n";
    }
}


TEST(MainTestSuite, TestIsClose) {
    EXPECT_TRUE(isClose(0.0001, 0.0, 1e-3, 1e-3));
    EXPECT_FALSE(isClose(0.01, 0.0, 1e-3, 1e-3));
    EXPECT_TRUE(isClose(1000.0, 1000.1, 1e-3, 1e-3));
    EXPECT_FALSE(isClose(1000.0, 1002.0, 1e-3, 1e-3));
}



TEST(MainTestSuite, TestFFTFloat) {
    size_t n_fft = 16;

    using float_t = float;
    using data_t = std::complex<float_t>;
    std::vector<data_t> series(n_fft);
    std::vector<data_t> out_cufft(n_fft);

    for (size_t i = 0; i < n_fft; ++i) {
        float_t v = static_cast<float_t>(i);
        series[i] = data_t(v, -v);
    }

    simpleFFT1d(series, out_cufft);

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
        EXPECT_TRUE(isClose(out_cufft[i].real(), out_fftw3[i].real(), float_t(1e-3), float_t(1e-5))) 
                        << "Vectors differ at [" << i << "]";
        EXPECT_TRUE(isClose(out_cufft[i].imag(), out_fftw3[i].imag(), float_t(1e-3), float_t(1e-5))) 
                        << "Vectors differ at [" << i << "]";
    }
}


// FFTW3 types differ for single/double precision
TEST(MainTestSuite, TestFFTDouble) {
    size_t n_fft = 16;

    using float_t = double;
    using data_t = std::complex<float_t>;
    std::vector<data_t> series(n_fft);
    std::vector<data_t> out_cufft(n_fft);

    for (size_t i = 0; i < n_fft; ++i) {
        float_t v = static_cast<float_t>(i);
        series[i] = data_t(v, -v);
    }

    simpleFFT1d(series, out_cufft);

    // Check against FFTW3
    std::vector<data_t> out_fftw3(n_fft);
    fftw_plan p;
    p = fftw_plan_dft_1d(
        n_fft,
        reinterpret_cast<fftw_complex*>(series.data()),
        reinterpret_cast<fftw_complex*>(out_fftw3.data()),
        FFTW_FORWARD, FFTW_ESTIMATE
    );
    fftw_execute(p);
    fftw_destroy_plan(p);

    for (size_t i = 0; i < n_fft; ++i) {
        EXPECT_TRUE(isClose(out_cufft[i].real(), out_fftw3[i].real(), float_t(1e-3), float_t(1e-5))) 
                        << "Vectors differ at [" << i << "]";
        EXPECT_TRUE(isClose(out_cufft[i].imag(), out_fftw3[i].imag(), float_t(1e-3), float_t(1e-5))) 
                        << "Vectors differ at [" << i << "]";
    }
}


TEST(MainTestSuite, TestMovingFFTFloat) {
    size_t n_series = 2048;
    size_t n_fft = 1024;
    size_t n_wins = n_series - n_fft + 1;
    size_t expected_out_size = n_wins * n_fft;

    using float_t = float;
    using data_t = std::complex<float_t>;
    std::vector<data_t> series(n_series);

    for (size_t i = 0; i < n_fft; ++i) {
        float_t v = static_cast<float_t>(i);
        series[i] = data_t(v, -v);
    }

    std::vector<data_t> out_cufft = movingFFT1d(series, n_fft);
    ASSERT_EQ(out_cufft.size(), expected_out_size) << "Output array has incorrect size";

    // Check against FFTW3
    std::vector<data_t> out_fftw3(expected_out_size);

    fftwf_plan p;
    p = fftwf_plan_dft_1d(n_fft, NULL, NULL, FFTW_FORWARD, FFTW_ESTIMATE);
    for (size_t i = 0; i < n_wins; ++i) {
        // Recreating plan is slow, but it is just to test FFT manually, without any complicated plans
        fftwf_execute_dft(
            p,
            reinterpret_cast<fftwf_complex*>(series.data() + i),
            reinterpret_cast<fftwf_complex*>(out_fftw3.data() + i * n_fft)
        );
    }
    fftwf_destroy_plan(p);

    for (size_t i = 0; i < expected_out_size; ++i) {
        EXPECT_TRUE(isClose(out_cufft[i].real(), out_fftw3[i].real(), float_t(7e-2), float_t(2e-2)))
                        << "Vectors differ at [" << i << "]";
        EXPECT_TRUE(isClose(out_cufft[i].imag(), out_fftw3[i].imag(), float_t(7e-2), float_t(2e-2)))
                        << "Vectors differ at [" << i << "]";
    }
}
